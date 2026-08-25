#!/usr/bin/env python3
"""Hold the compiled marker corpus to the assertions its brief names.

`canonical/owner_authored_marker_corpus.md` lists the conditions under which a
compiled arm set must fail. This file is those conditions, plus the one thing
the brief asks for that no existing check can observe -- semantic arm drift from
a single scene specification.

**Most of these cannot fail while the compiler is correct, and that is the
point of running them anyway.** `compile_marker_corpus.py` derives every arm
from one serialization, so arm drift is unrepresentable rather than merely
unlikely. This file is what says so from the outside: it re-derives each arm's
text from the canonical prompt without reading the compiler's transformation
code, and compares. If someone later gives an arm its own prose, the derivation
stops matching.

The arms:

1. **Arm derivation.** Each arm's prompt must be exactly the declared transform
   of the canonical text, recomputed here.
2. **Ordinary-text alignment.** Strip the markers from any arm and the
   remainder must hash identically across all four. The brief requires an
   explicit alignment map; this is that claim in a form that can be false.
3. **Released ids.** The seven markers must resolve to 151669--151675, read
   from the emitted record rather than from a constant this file also owns.
4. **A genuine legacy arm.** The legacy tokenizer must not own the markers, and
   its token stream must differ from the release arm's on marker-bearing rows.
   A legacy arm that silently became a second copy of the release arm is the
   escaped defect `audit_h3_marker_tokenization` was repaired for.
5. **Stripped enclosure.** Every paired marker's enclosed text must survive the
   stripped arm byte for byte.
6. **Marker grammar.** Delegated to `preflight_graph.marker_rules`, which
   already owns the five-marker nesting rules. The brief says to run the
   existing checks rather than invent a parallel grammar.
7. **Media reality.** Every declared media file must exist at the pinned
   snapshot, hash to what the scene declares, and be a genuine pooled input
   reference rather than a generated output.
8. **Split hygiene.** No prompt hash and no media hash may appear in two splits.
9. **Model transforms are declarations.** No arm may claim a model transform
   this repo applies, and nothing here may write a token row.

`--violation-arm` mutates a compiled corpus and requires each named arm to gain
a problem the unmutated baseline did not have. Per `docs/checks.md`, a mutation
control needs its own precondition: each mutation here raises rather than
passing when the corpus carries no eligible subject.

CPU only, no CUDA, no weights, no server.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

from compile_marker_corpus import (  # noqa: E402
    ALL_MARKERS,
    ARMS,
    CAPTION,
    DIALOGUE,
    LYRICS,
    RELEASE_MARKER_IDS,
    sha,
    strip_markers,
)

PAIRS = (DIALOGUE, CAPTION, LYRICS)
DEFAULT = BENCH / "marker_corpus" / "compiled.json"


def enclosed(text: str, pair: tuple[str, str]) -> list[str]:
    """Text between each open/close pair, in order, non-greedy."""
    out, start = [], 0
    while True:
        left = text.find(pair[0], start)
        if left < 0:
            return out
        right = text.find(pair[1], left + len(pair[0]))
        if right < 0:
            return out
        out.append(text[left + len(pair[0]):right])
        start = right + len(pair[1])


def arm_derivation(scene: dict) -> list[str]:
    """Each arm is the declared transform of the canonical text, recomputed."""
    problems = []
    canonical = scene["canonical_prompt"]
    if sha(canonical) != scene["canonical_prompt_sha256"]:
        problems.append(f"{scene['spec_id']}: canonical prompt does not match its hash")
    expected = {
        "release_id": canonical,
        "legacy_bpe": canonical,
        "stripped": strip_markers(canonical),
        "mean_init_rows": canonical,
    }
    for name in ARMS:
        arm = scene["arms"].get(name)
        if arm is None:
            problems.append(f"{scene['spec_id']}: arm {name} is absent")
            continue
        if arm["prompt"] != expected[name]:
            problems.append(
                f"{scene['spec_id']} {name}: prompt is not the declared "
                f"transform of the canonical scene text")
        if sha(arm["prompt"]) != arm["prompt_sha256"]:
            problems.append(f"{scene['spec_id']} {name}: prompt hash disagrees")
        if len(arm["prompt"].encode()) != arm["prompt_bytes"]:
            problems.append(f"{scene['spec_id']} {name}: prompt_bytes disagrees")
    return problems


def arm_alignment(scene: dict) -> list[str]:
    """The ordinary text must be one string across every arm."""
    problems = []
    hashes = {name: sha(strip_markers(arm["prompt"]))
              for name, arm in scene["arms"].items()}
    if len(set(hashes.values())) != 1:
        problems.append(
            f"{scene['spec_id']}: ordinary text differs across arms {hashes}; "
            f"the alignment map is claimed where none exists")
    for name, arm in scene["arms"].items():
        if arm["ordinary_text_sha256"] != hashes[name]:
            problems.append(
                f"{scene['spec_id']} {name}: recorded ordinary_text_sha256 is "
                f"not the hash of its own stripped prompt")
    if scene["ordinary_text_sha256"] != next(iter(hashes.values())):
        problems.append(
            f"{scene['spec_id']}: the scene's ordinary_text_sha256 is not the "
            f"arms' shared value")
    return problems


def arm_release_ids(scene: dict) -> list[str]:
    problems = []
    for name, arm in scene["arms"].items():
        for span in arm["marker_spans"]:
            declared = span["release_id"]
            if declared != RELEASE_MARKER_IDS[span["marker"]]:
                problems.append(
                    f"{scene['spec_id']} {name}: {span['marker']} recorded as "
                    f"{declared}, the release declares "
                    f"{RELEASE_MARKER_IDS[span['marker']]}")
            if not 151669 <= declared <= 151675:
                problems.append(
                    f"{scene['spec_id']} {name}: {span['marker']} id {declared} "
                    f"is outside the released range")
        tokens = arm.get("tokens", {})
        if not tokens.get("recorded"):
            continue
        if name == "stripped" and tokens["marker_ids_present"]:
            problems.append(
                f"{scene['spec_id']} stripped: marker ids "
                f"{tokens['marker_ids_present']} survive the stripped arm")
        if name in ("release_id", "mean_init_rows") and arm["marker_spans"] \
                and not tokens["marker_ids_present"]:
            problems.append(
                f"{scene['spec_id']} {name}: the text carries markers but no "
                f"marker id reached the token stream")
    return problems


def arm_legacy_is_genuine(scene: dict) -> list[str]:
    """A legacy arm that became a second copy of the release arm is the defect."""
    problems = []
    release = scene["arms"]["release_id"].get("tokens", {})
    legacy = scene["arms"]["legacy_bpe"].get("tokens", {})
    if not (release.get("recorded") and legacy.get("recorded")):
        return problems
    if legacy["tokenizer"] != "legacy":
        problems.append(f"{scene['spec_id']}: the legacy arm records a "
                        f"{legacy['tokenizer']!r} tokenizer")
    if not scene["arms"]["release_id"]["marker_spans"]:
        return problems
    if legacy["marker_ids_present"]:
        problems.append(
            f"{scene['spec_id']}: the legacy arm resolved marker ids "
            f"{legacy['marker_ids_present']}, so it is not unpatched")
    if legacy["ids_sha256"] == release["ids_sha256"]:
        problems.append(
            f"{scene['spec_id']}: legacy and release token streams are "
            f"identical on a marker-bearing prompt, so the legacy arm is a "
            f"second copy of the release arm and every delta against it is zero "
            f"for the wrong reason")
    return problems


def arm_stripped_enclosure(scene: dict) -> list[str]:
    """Removing a marker must not touch the text it wrapped."""
    problems = []
    canonical = scene["canonical_prompt"]
    stripped = scene["arms"]["stripped"]["prompt"]
    for pair in PAIRS:
        for body in enclosed(canonical, pair):
            bare = strip_markers(body)
            if bare and bare not in stripped:
                problems.append(
                    f"{scene['spec_id']}: text enclosed by {pair[0]} did not "
                    f"survive the stripped arm: {bare[:60]!r}")
    for marker in ALL_MARKERS:
        if marker in stripped:
            problems.append(
                f"{scene['spec_id']}: {marker} survives the stripped arm")
    return problems


def arm_marker_grammar(scene: dict) -> list[str]:
    """Delegated to the grammar the repo already owns."""
    import importlib.util

    path = BENCH / "preflight_graph.py"
    spec = importlib.util.spec_from_file_location("_preflight", path)
    if spec is None or spec.loader is None:
        return [f"cannot load {path.name} to reuse its marker grammar"]
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    problems = []
    prompt = scene["canonical_prompt"]
    body = prompt.split("detailed_description:", 1)[-1]
    for level, message in module.marker_rules(prompt, body):
        if level == "FAIL":
            problems.append(f"{scene['spec_id']}: marker grammar: {message}")
    return problems


def arm_media_reality(scene: dict, root: Path, pool_media: dict) -> list[str]:
    problems = []
    for item in scene["media"]:
        if item["role"] == "reference-audio":
            continue
        path = root / item["path"]
        if not path.is_file():
            problems.append(
                f"{scene['spec_id']} {item['label']}: {item['path']} is not in "
                f"the pinned snapshot")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != item["sha256"]:
            problems.append(
                f"{scene['spec_id']} {item['label']}: declares "
                f"{item['sha256'][:12]}, the snapshot is {actual[:12]}")
        if item["path"] not in pool_media:
            problems.append(
                f"{scene['spec_id']} {item['label']}: {item['path']} is not a "
                f"declared input reference of the accepted pool, so it cannot "
                f"be shown to be a genuine reference rather than generated "
                f"output")
    return problems


def arm_split_hygiene(scenes: list[dict]) -> list[str]:
    problems = []
    by_split: dict[str, set] = {}
    media_by_split: dict[str, set] = {}
    for scene in scenes:
        by_split.setdefault(scene["split"], set()).add(
            scene["canonical_prompt_sha256"])
        media_by_split.setdefault(scene["split"], set()).update(
            m["sha256"] for m in scene["media"] if m.get("sha256"))
    splits = sorted(by_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            shared = by_split[a] & by_split[b]
            if shared:
                problems.append(f"prompt hash shared by splits {a} and {b}: {shared}")
            shared_media = media_by_split[a] & media_by_split[b]
            if shared_media:
                problems.append(
                    f"media hash shared by splits {a} and {b}: {sorted(shared_media)}")
    seen: dict[str, str] = {}
    for scene in scenes:
        digest = scene["canonical_prompt_sha256"]
        if digest in seen and seen[digest] != scene["spec_id"]:
            problems.append(
                f"{scene['spec_id']} and {seen[digest]} compile to the same "
                f"prompt; two scene specifications produced one scene")
        seen[digest] = scene["spec_id"]
    return problems


def arm_transforms_are_declarations(document: dict) -> list[str]:
    problems = []
    for name, transform in document["arm_transforms"].items():
        model = transform["model"]
        if model != "none" and "DECLARED ONLY" not in model:
            problems.append(
                f"arm {name} claims a model transform without marking it a "
                f"declaration: {model!r}")
    for scene in document["scenes"]:
        for name, arm in scene["arms"].items():
            if arm["transform"] != document["arm_transforms"][name]:
                problems.append(
                    f"{scene['spec_id']} {name}: arm-level transform differs "
                    f"from the corpus declaration")
    return problems


# ---------------------------------------------------------------------------
# violation arm


MUTATIONS = {
    "arm-prose-drift": ("arm_derivation",
                        "one arm gets its own prose instead of the transform"),
    "arm-alignment-break": ("arm_alignment",
                            "an arm's ordinary text stops matching the others"),
    "wrong-release-id": ("arm_release_ids",
                         "a marker is recorded outside the released id range"),
    "legacy-is-release": ("arm_legacy_is_genuine",
                          "the legacy arm silently becomes a second release arm"),
    "stripped-eats-text": ("arm_stripped_enclosure",
                           "stripping a marker also removes the text it wrapped"),
    "bad-nesting": ("arm_marker_grammar",
                    "a caption pair is made to wrap a <d> block"),
    "wrong-media-hash": ("arm_media_reality",
                         "a scene declares a media digest the snapshot does not produce"),
    "duplicate-scene": ("arm_split_hygiene",
                        "two scene specifications compile to the same prompt"),
    "undeclared-model-transform": ("arm_transforms_are_declarations",
                                   "an arm claims a model transform as applied"),
}


def _mutate(document: dict, kind: str) -> None:
    scenes = document["scenes"]

    def with_markers():
        for scene in scenes:
            if scene["arms"]["release_id"]["marker_spans"]:
                return scene
        raise LookupError("no scene carries a marker")

    def with_media():
        for scene in scenes:
            if scene["media"]:
                return scene
        raise LookupError("no scene carries media")

    def with_pair(pair):
        for scene in scenes:
            if pair[0] in scene["canonical_prompt"]:
                return scene
        raise LookupError(f"no scene uses {pair[0]}")

    if kind == "arm-prose-drift":
        scene = with_markers()
        scene["arms"]["stripped"]["prompt"] += " The camera then cuts away."
        scene["arms"]["stripped"]["prompt_sha256"] = sha(
            scene["arms"]["stripped"]["prompt"])
        scene["arms"]["stripped"]["prompt_bytes"] = len(
            scene["arms"]["stripped"]["prompt"].encode())
    elif kind == "arm-alignment-break":
        scene = with_markers()
        arm = scene["arms"]["mean_init_rows"]
        arm["prompt"] = arm["prompt"].replace("the", "teh", 1)
        arm["prompt_sha256"] = sha(arm["prompt"])
        arm["prompt_bytes"] = len(arm["prompt"].encode())
        arm["ordinary_text_sha256"] = sha(strip_markers(arm["prompt"]))
    elif kind == "wrong-release-id":
        scene = with_markers()
        scene["arms"]["release_id"]["marker_spans"][0]["release_id"] = 151668
    elif kind == "legacy-is-release":
        scene = with_markers()
        scene["arms"]["legacy_bpe"]["tokens"] = copy.deepcopy(
            scene["arms"]["release_id"]["tokens"])
        scene["arms"]["legacy_bpe"]["tokens"]["tokenizer"] = "legacy"
    elif kind == "stripped-eats-text":
        scene = with_pair(DIALOGUE)
        body = enclosed(scene["canonical_prompt"], DIALOGUE)[0]
        arm = scene["arms"]["stripped"]
        arm["prompt"] = arm["prompt"].replace(strip_markers(body), "", 1)
        arm["prompt_sha256"] = sha(arm["prompt"])
    elif kind == "bad-nesting":
        scene = with_pair(DIALOGUE)
        scene["canonical_prompt"] = scene["canonical_prompt"].replace(
            DIALOGUE[0], CAPTION[0] + DIALOGUE[0], 1).replace(
            DIALOGUE[1], DIALOGUE[1] + CAPTION[1], 1)
        scene["canonical_prompt_sha256"] = sha(scene["canonical_prompt"])
    elif kind == "wrong-media-hash":
        scene = with_media()
        scene["media"][0]["sha256"] = "0" * 64
    elif kind == "duplicate-scene":
        if len(scenes) < 2:
            raise LookupError("need two scenes to duplicate one")
        scenes[1]["canonical_prompt_sha256"] = scenes[0]["canonical_prompt_sha256"]
    elif kind == "undeclared-model-transform":
        document["arm_transforms"]["mean_init_rows"]["model"] = (
            "replace the seven marker embedding rows with the table mean")
        for scene in scenes:
            scene["arms"]["mean_init_rows"]["transform"] = \
                document["arm_transforms"]["mean_init_rows"]
    else:
        raise ValueError(kind)


def evaluate(document: dict, root: Path, pool_media: dict) -> dict:
    scenes = document["scenes"]
    problems: dict[str, list[str]] = {
        "arm_derivation": [], "arm_alignment": [], "arm_release_ids": [],
        "arm_legacy_is_genuine": [], "arm_stripped_enclosure": [],
        "arm_marker_grammar": [], "arm_media_reality": [],
        "arm_split_hygiene": [], "arm_transforms_are_declarations": [],
    }
    for scene in scenes:
        problems["arm_derivation"] += arm_derivation(scene)
        problems["arm_alignment"] += arm_alignment(scene)
        problems["arm_release_ids"] += arm_release_ids(scene)
        problems["arm_legacy_is_genuine"] += arm_legacy_is_genuine(scene)
        problems["arm_stripped_enclosure"] += arm_stripped_enclosure(scene)
        problems["arm_marker_grammar"] += arm_marker_grammar(scene)
        problems["arm_media_reality"] += arm_media_reality(scene, root, pool_media)
    problems["arm_split_hygiene"] += arm_split_hygiene(scenes)
    problems["arm_transforms_are_declarations"] += \
        arm_transforms_are_declarations(document)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, default=DEFAULT)
    parser.add_argument("--violation-arm", action="store_true")
    args = parser.parse_args()

    if not args.compiled.is_file():
        print(f"{args.compiled} is absent; run compile_marker_corpus.py first")
        return 2

    from build_h3_calibration_pool import pinned_snapshot

    root, _ = pinned_snapshot()
    pool_media: dict[str, str] = {}
    pool_path = BENCH / "results" / "2026-08-24_h3_calibration_pool.jsonl"
    for line in pool_path.read_text().splitlines():
        pool_media.update(json.loads(line).get("media_sha256") or {})

    document = json.loads(args.compiled.read_text())

    if args.violation_arm:
        baseline = evaluate(document, root, pool_media)
        escaped = []
        for kind, (arm, description) in MUTATIONS.items():
            mutated = copy.deepcopy(document)
            try:
                _mutate(mutated, kind)
            except LookupError as exc:
                escaped.append(f"{kind}: could not be applied ({exc})")
                print(f"[ESCAPED] {arm} <- {kind}: {description}")
                continue
            after = evaluate(mutated, root, pool_media)
            gained = [p for p in after[arm] if p not in baseline[arm]]
            status = "caught" if gained else "ESCAPED"
            if not gained:
                escaped.append(f"{kind}: {arm} gained no problem")
            print(f"[{status}] {arm} <- {kind}: {description}")
        print(f"\nmutations that escaped: {len(escaped)} of {len(MUTATIONS)}")
        return 1 if escaped else 0

    problems = evaluate(document, root, pool_media)
    total = sum(len(v) for v in problems.values())
    for arm, items in sorted(problems.items()):
        print(f"[{'GREEN' if not items else f'RED ({len(items)})'}] {arm}")
        for item in items[:10]:
            print(f"    - {item}")
        if len(items) > 10:
            print(f"    ... {len(items) - 10} more")
    scenes = document["scenes"]
    print(f"\n{len(scenes)} scene(s), {len(ARMS)} arms each, "
          f"{sum(len(s['media']) for s in scenes)} media references")
    for cell in document.get("declared_missing_cells", []):
        print(f"  declared missing: {cell['cell']}")
    print(f"blocking problems: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
