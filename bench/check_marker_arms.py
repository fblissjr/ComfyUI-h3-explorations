#!/usr/bin/env python3
"""A marker-corpus arm that was declared and not applied must go red.

`marker_arms.py` binds one arm of `bench/marker_corpus/compiled.json` to a CLIP
and records what bound. The failure this guards is narrow and quiet: an arm
node in the graph, an arm name in the record, and the render actually produced
under the release tokenizer and the release rows. Nothing about the clip, the
sampler or the output would look wrong, and the sidecar would say the arm ran.

So every case here has one shape -- **the declaration and the value read back
off the live CLIP must agree** -- which is
`bench/check_provenance_stamp.py::closure_is_read_not_declared` generalised
from the Sol closure to the encoder.

## What it runs against, and why that is not the real encoder

The row cases need a real `ModelPatcher`, real `add_patches`, and ComfyUI's
real `calculate_weight`. They do not need the real weights, so the fixture is
a synthetic embedding with the real vocabulary height and a narrow width --
megabytes, no artifact, no CUDA, no server. The tokenizer cases use the real
`MiniMaxH3Tokenizer`, which is small.

**A fixture built to satisfy this module's own key constant would prove
nothing about the artifact**, so `embed_key_is_the_real_one` re-reads the key
off the installed encoder and skips loudly when it is absent. That case is the
reason the synthetic fixture is honest.

## The cross-implementation case

`live_tokenizers_reproduce_the_corpus` is the one that is not self-referential.
The corpus recorded its own token ids for the release and legacy arms, from a
compiler that shares no code with `marker_arms.py`. Reproducing those ids with
the reconstruction this repo will actually render under is an agreement between
two implementations, which is worth more than any assertion here against
numbers this file computed itself.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parents[1]
sys.path.insert(0, str(COMFY))

import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True

CORPUS = REPO / "bench" / "marker_corpus" / "compiled.json"
ARTIFACT = COMFY / "models" / "text_encoders" / "qwen3vl_32b_minimax_h3_w4a16_awq.safetensors"
# Narrow, so the fixture is megabytes. The HEIGHT is real: the marker ids are
# absolute positions in the vocabulary and a short table cannot address them.
FIXTURE_WIDTH = 8


def _module(name: str, path: Path):
    pkg = types.ModuleType("_h3marker")
    pkg.__path__ = [str(REPO)]
    sys.modules["_h3marker"] = pkg
    spec = importlib.util.spec_from_file_location(f"_h3marker.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _module("marker_arms", REPO / "marker_arms.py")


def _fixture_clip(width: int = FIXTURE_WIDTH):
    """A real CLIP object over a synthetic embedding at the real key.

    Built through `comfy.sd.CLIP(no_init=True)` and populated the way
    `CLIP.clone()` populates a clone, so `clone`, `add_patches` and `tokenize`
    are the real methods and not stand-ins.
    """
    import torch
    import comfy.model_patcher
    import comfy.sd
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer

    start, count = M.marker_row_span()
    rows = start + count + 260  # past the markers, as the real vocabulary is

    # Nested so the state-dict key is exactly `marker_arms.EMBED_KEY`.
    embed = torch.nn.Embedding(rows, width, dtype=torch.bfloat16)
    with torch.no_grad():
        embed.weight.copy_(torch.arange(rows, dtype=torch.float32)
                           .unsqueeze(1).expand(rows, width) * 1e-3)
    inner = torch.nn.Module(); inner.embed_tokens = embed
    transformer = torch.nn.Module(); transformer.model = inner
    qwen = torch.nn.Module(); qwen.transformer = transformer
    root = torch.nn.Module(); root.qwen3vl_32b = qwen
    assert M.EMBED_KEY in root.state_dict(), sorted(root.state_dict())

    clip = comfy.sd.CLIP(no_init=True)
    clip.cond_stage_model = root
    clip.patcher = comfy.model_patcher.ModelPatcher(
        root, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    clip.tokenizer = MiniMaxH3Tokenizer()
    clip.layer_idx = None
    clip.tokenizer_options = {}
    clip.use_clip_schedule = False
    clip.apply_hooks_to_conds = None
    return clip


def _scenes():
    corpus = json.loads(CORPUS.read_text())
    return corpus, corpus.get("scenes") or []


# --------------------------------------------------------------------------
# 1 and 2: the prompt-bytes arms. No CLIP involved -- these arms are text.


def corpus_prompt_bytes_are_what_they_declare():
    _, scenes = _scenes()
    assert scenes, "the corpus declares no scenes; nothing to check"
    markers = M.marker_tokens()
    checked = 0
    for scene in scenes:
        for arm, body in (scene.get("arms") or {}).items():
            text = body["prompt"]
            digest = hashlib.sha256(text.encode()).hexdigest()
            assert digest == body["prompt_sha256"], (scene["spec_id"], arm)
            if arm == "stripped":
                # Stripping removes the marker strings and nothing else, so the
                # stripped prompt IS the ordinary text and carries no marker.
                assert body["prompt_sha256"] == body["ordinary_text_sha256"], (
                    scene["spec_id"], "stripped prompt is not the ordinary text")
                present = [m for m in markers if m in text]
                assert not present, (scene["spec_id"], f"markers survived: {present}")
                assert not body["marker_spans"], scene["spec_id"]
            else:
                assert digest == scene["canonical_prompt_sha256"], (
                    scene["spec_id"], arm, "arm text diverged from the canonical prompt")
            checked += 1
    assert checked, "no arm carried a prompt"
    return f"{checked} arm prompt(s) across {len(scenes)} scene(s)"


def edited_prompt_bytes_are_refused():
    """Red control for both text arms: any edit away from the corpus fails."""
    _, scenes = _scenes()
    markers = M.marker_tokens()
    scene = scenes[0]

    # release_id: one character changed.
    body = copy.deepcopy(scene["arms"]["release_id"])
    body["prompt"] = body["prompt"].replace("the", "teh", 1)
    assert hashlib.sha256(body["prompt"].encode()).hexdigest() != body["prompt_sha256"]

    # stripped: a marker left in. Its digest moves AND the marker is findable,
    # so the arm fails whether or not anyone recomputed the digest.
    stripped = copy.deepcopy(scene["arms"]["stripped"])
    leaked = stripped["prompt"] + markers[0]
    assert hashlib.sha256(leaked.encode()).hexdigest() != stripped["prompt_sha256"]
    assert any(m in leaked for m in markers)

    # stripped: ordinary text altered while stripping correctly.
    moved = stripped["prompt"].replace("the", "teh", 1)
    assert hashlib.sha256(moved.encode()).hexdigest() != stripped["ordinary_text_sha256"]
    return "edited release_id, leaked marker, and moved ordinary text all refused"


# --------------------------------------------------------------------------
# The cross-implementation case.


def live_tokenizers_reproduce_the_corpus():
    _, scenes = _scenes()
    release = _fixture_clip()
    legacy = M.legacy_tokenizer_clip(release)
    seen = {"release": 0, "legacy": 0}
    for scene in scenes:
        if scene.get("tokenization_skipped_because"):
            continue
        for arm, body in (scene.get("arms") or {}).items():
            tokens = body.get("tokens") or {}
            if not tokens.get("recorded") or "ids" not in tokens:
                continue
            which = tokens.get("tokenizer")
            clip = {"release": release, "legacy": legacy}.get(which)
            if clip is None:
                raise AssertionError(f"{scene['spec_id']}/{arm}: unknown "
                                     f"tokenizer {which!r} in the corpus")
            got = [t[0] for t in clip.tokenize(body["prompt"])["qwen3vl_32b"][0]]
            assert got == list(tokens["ids"]), (
                f"{scene['spec_id']}/{arm}: this repo's {which} tokenizer does "
                f"not reproduce the corpus ids ({len(got)} vs "
                f"{len(tokens['ids'])} tokens)")
            seen[which] += 1
    assert seen["release"] and seen["legacy"], (
        f"only one tokenizer was exercised {seen}; this case cannot show the "
        "two arms disagree")
    return f"{seen['release']} release and {seen['legacy']} legacy arm(s) reproduced"


# --------------------------------------------------------------------------
# 3 and 4: the tokenizer arm.


def declaring_legacy_does_not_make_a_clip_legacy():
    """Red control: the label says legacy, the tokenizer is the release one."""
    clip = _fixture_clip()
    honest = M.apply_arm(clip, "legacy_bpe")
    liar = clip.clone()
    liar._h3_declared_marker_arm = "legacy_bpe"   # label only, nothing bound

    truth = M.encoder_arm_record(honest)["tokenizer"]
    lie = M.encoder_arm_record(liar)["tokenizer"]
    release = M.encoder_arm_record(clip)["tokenizer"]

    assert lie["probe_sha256"] == release["probe_sha256"], (
        "the fixture's unbound clip already differs from the release tokenizer")
    assert truth["probe_sha256"] != release["probe_sha256"], (
        "the legacy arm did not change the probe tokenization")
    assert truth["markers_resolved"] == 0, truth["markers_resolved"]
    assert lie["markers_resolved"] == len(M.marker_tokens()), (
        f"a clip carrying only the legacy LABEL reported "
        f"{lie['markers_resolved']} markers resolved; the record is reading "
        "declared_arm instead of the tokenizer")
    return (f"declared-only: {lie['markers_resolved']} markers still resolve; "
            f"bound: {truth['markers_resolved']}")


def legacy_reconstruction_refuses_when_it_can_empty_nothing():
    """Red control: the reconstruction keyed off a name that is not there.

    The escaped instance behind this is on the record -- a harness keyed off
    the constant's NAME returned the corrected tokenizer labelled stock when
    upstream renamed it. Here the two known names are removed, so a version
    that shrugged would hand back the release tokenizer under a legacy label.
    It must raise instead.
    """
    import comfy.text_encoders.minimax as mmx
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer

    clip = _fixture_clip()
    saved = [(owner, attr, getattr(owner, attr, None))
             for owner, attr in ((MiniMaxH3Tokenizer, "H3_SPECIAL_TOKENS"),
                                 (mmx, "MINIMAX_EXTRA_TOKENS"))]
    try:
        for owner, attr, _ in saved:
            if hasattr(owner, attr):
                delattr(owner, attr)
        try:
            M.legacy_tokenizer_clip(clip)
        except ValueError as exc:
            assert "cannot be reconstructed" in str(exc), exc
        else:
            raise AssertionError(
                "with no known token list present the reconstruction returned "
                "a clip instead of refusing; that clip is the release tokenizer")
    finally:
        for owner, attr, value in saved:
            if value is not None:
                setattr(owner, attr, value)
    # And the real attribute is restored, so nothing later in this run is
    # measuring a tokenizer this case broke.
    assert M.encoder_arm_record(_fixture_clip())["tokenizer"]["markers_resolved"] \
        == len(M.marker_tokens())
    return "refuses, and the tokenizer is intact afterwards"


# --------------------------------------------------------------------------
# 5 and 6: the row arm.


def row_transform_attaches_or_refuses():
    """Red control: a patch keyed to something the model does not have."""
    import torch

    clip = _fixture_clip()
    before = M.encoder_arm_record(clip)["marker_rows"]["sha256"]

    bound = M.apply_arm(clip, "mean_init_rows")
    after = M.encoder_arm_record(bound)["marker_rows"]
    assert after["sha256"] != before, "the row transform did not move the rows"
    assert after["patch_keys"] == [M.EMBED_KEY], after["patch_keys"]

    # The violation: the same transform aimed at a key the model lacks. Nothing
    # attaches, and the rows are the release ones under a mean_init label.
    start, count = M.marker_row_span()
    wrong = clip.clone()
    matched = wrong.add_patches(
        {("qwen3vl_32b.transformer.model.embed_tokens.WEIGHT", (0, start, count)):
            ("set", (torch.zeros(count, FIXTURE_WIDTH, dtype=torch.bfloat16),))},
        1.0, 1.0)
    wrong._h3_declared_marker_arm = "mean_init_rows"
    record = M.encoder_arm_record(wrong)["marker_rows"]
    assert not matched, f"a wrong key matched: {matched}"
    assert record["sha256"] == before, (
        "the rows moved without a patch attaching, so this digest is not "
        "reading the patcher")
    assert record["patch_keys"] == []
    return f"bound moves the rows; a wrong key attaches nothing and stays at {before[:12]}"


def the_patch_route_does_not_reach_the_loaded_model():
    """Red control: the in-place write the patch route exists to avoid.

    `CLIP.clone()` shares `cond_stage_model`, and the encoder loader caches its
    patcher, so an in-place assignment is inherited by every later render that
    reuses the model. The patched arm must leave the original untouched; the
    in-place version must not, which is what makes this case able to fail.
    """
    import torch

    clip = _fixture_clip()
    start, count = M.marker_row_span()
    original = M.encoder_arm_record(clip)["marker_rows"]["sha256"]

    bound = M.apply_arm(clip, "mean_init_rows")
    assert M.encoder_arm_record(bound)["marker_rows"]["sha256"] != original, (
        "the bound arm's rows equal the unbound ones, so this case cannot go "
        "on to show whether the original leaked")
    assert M.encoder_arm_record(clip)["marker_rows"]["sha256"] == original, (
        "the patch route reached the original CLIP")

    # The violation, on a throwaway fixture so nothing later sees it.
    leaky = _fixture_clip()
    leaky_before = M.encoder_arm_record(leaky)["marker_rows"]["sha256"]
    sibling = leaky.clone()
    with torch.no_grad():
        weight = leaky.patcher.model.state_dict()[M.EMBED_KEY]
        weight[start:start + count] = 0
    assert M.encoder_arm_record(sibling)["marker_rows"]["sha256"] != leaky_before, (
        "an in-place write did NOT reach a sibling clone, so this fixture "
        "cannot show the leak the patch route avoids")
    return "patched arm is isolated; the in-place write reaches a sibling clone"


# --------------------------------------------------------------------------
# 7: the case that governs the others.


def different_transforms_do_not_share_a_record():
    """Red control: a record that reads the label instead of the CLIP.

    A stamp recording what it was told is indistinguishable from one recording
    what happened, until two arms that should differ do not. All three arms are
    built from ONE fixture, so anything shared between their records is shared
    because nothing read the CLIP.
    """
    clip = _fixture_clip()
    records = {arm: M.encoder_arm_record(M.apply_arm(clip, arm)) for arm in M.ARMS}

    for arm, record in records.items():
        assert record["declared_arm"] == arm, record["declared_arm"]

    blobs = {arm: json.dumps({k: v for k, v in r.items() if k != "declared_arm"},
                             sort_keys=True)
             for arm, r in records.items()}
    collisions = [(a, b) for a in M.ARMS for b in M.ARMS
                  if a < b and blobs[a] == blobs[b]]
    assert not collisions, (
        f"arms produced identical records once the label is removed: "
        f"{collisions}. The record is reading the declaration, not the CLIP.")

    # And each arm moves the part it claims, not some other part.
    assert records["legacy_bpe"]["tokenizer"] != records["release"]["tokenizer"]
    assert records["legacy_bpe"]["marker_rows"] == records["release"]["marker_rows"]
    assert records["mean_init_rows"]["marker_rows"] != records["release"]["marker_rows"]
    assert records["mean_init_rows"]["tokenizer"] == records["release"]["tokenizer"]
    return f"{len(M.ARMS)} arms, {len(set(blobs.values()))} distinct records"


# --------------------------------------------------------------------------


def embed_key_is_the_real_one(path: Path):
    """The constant the fixture is built around, re-read from the artifact.

    Without this the fixture proves only that this module agrees with itself.
    """
    module = _module("h3_awq_encoder", REPO / "h3_awq_encoder.py")
    clip = module._load_clip(str(path), [], device="cpu")
    state = clip.patcher.model.state_dict()
    assert M.EMBED_KEY in state, (
        f"{M.EMBED_KEY} is not in the real encoder; the fixture is built "
        f"around a key that does not exist. Present: "
        f"{[k for k in state if 'embed' in k]}")
    rows = int(state[M.EMBED_KEY].shape[0])
    start, count = M.marker_row_span(rows)
    assert start + count <= rows
    record = M.encoder_arm_record(clip)
    assert record["tokenizer"]["markers_resolved"] == len(M.marker_tokens())
    return f"{M.EMBED_KEY} present, {rows} rows, markers at {start}..{start + count - 1}"


def provenance_records_three_states():
    """No CLIP, an unarmed CLIP, and an armed one are distinguishable.

    The repo's rule for anything that gains an absent state: every assertion
    about it inherits a third case, and "correctly absent" is not "broken".
    """
    prov = _module("provenance", REPO / "provenance.py")
    stamp = prov.MiniMaxH3ProvenanceStamp

    assert stamp._encoder_arm(None) == prov.NOT_DETECTED

    clip = _fixture_clip()
    unarmed = stamp._encoder_arm(clip)
    assert unarmed["declared_arm"] is None, unarmed["declared_arm"]
    assert unarmed["tokenizer"]["markers_resolved"] == len(M.marker_tokens())

    armed = stamp._encoder_arm(M.apply_arm(clip, "legacy_bpe"))
    assert armed["declared_arm"] == "legacy_bpe"
    assert armed["tokenizer"]["probe_sha256"] != \
        unarmed["tokenizer"]["probe_sha256"], (
            "the armed and unarmed states stamp the same tokenizer digest, so "
            "a sidecar cannot tell an arm that ran from one that did not")

    # The CLIP input is LAST in the schema. A saved graph matches widget values
    # by index, so this is the difference between adding an input and silently
    # re-pointing every existing provenance node.
    names = [getattr(i, "id", None) for i in stamp.define_schema().inputs]
    assert names[-1] == "clip", names
    return "absent / unarmed / armed all distinct, and the input is appended"


def main() -> int:
    cases = [
        ("corpus prompt bytes", corpus_prompt_bytes_are_what_they_declare),
        ("edited prompt refused", edited_prompt_bytes_are_refused),
        ("corpus token ids reproduced", live_tokenizers_reproduce_the_corpus),
        ("declared legacy is not legacy", declaring_legacy_does_not_make_a_clip_legacy),
        ("legacy refuses to no-op", legacy_reconstruction_refuses_when_it_can_empty_nothing),
        ("row transform attaches", row_transform_attaches_or_refuses),
        ("patch route does not leak", the_patch_route_does_not_reach_the_loaded_model),
        ("arms do not share a record", different_transforms_do_not_share_a_record),
        ("provenance three states", provenance_records_three_states),
    ]
    raw = os.environ.get("H3_AWQ_ENCODER")
    artifact = Path(os.path.expanduser(raw)) if raw else ARTIFACT
    if artifact.exists():
        cases.append(("real embedding key", lambda: embed_key_is_the_real_one(artifact)))
    else:
        print(f"  SKIP  real embedding key: no encoder at {artifact.name}; the "
              "synthetic fixture is then only self-consistent")

    if not CORPUS.exists():
        print(f"SKIP  marker corpus absent at {CORPUS.relative_to(REPO)}")
        return 2

    ok = True
    for label, case in cases:
        try:
            detail = case()
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {label}: {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
