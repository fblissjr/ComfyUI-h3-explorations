#!/usr/bin/env python3
"""The ordered-reference resolver against the socket resolver it replaces.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). **It imports
ComfyUI** -- core is the oracle, so it must -- with stub VAEs and a patched
tokenizer. No CUDA, no model, no server.

**Two verdicts, not one, and conflating them would reject a correct
replacement.** `docs/research/conditioning_nodes.md` splits the acceptance
criteria into behaviour a replacement must PRESERVE and behaviour it
intentionally REPLACES. A suite asserting AGREE on all of it would fail the
new model for doing its job, so:

  AGREE   socket-shaped inputs must produce core's own two lists exactly,
          over a CHOSEN set of configurations rather than a sweep -- driving
          core costs a node execution each, so the set is picked to include
          the sparse and both-sounded cases a generator never emits, and the
          count is printed so a shrinking set is visible. This is the claim
          that the ordered model is a superset and not a rewrite.

  DIFFER  the two replaced behaviours must produce labels the socket model
          could not. If these ever AGREE, the ordered model gained nothing
          and the surface is not worth its cost.

**CORE is the control, not `wired_labels`.** The legacy authority is what
`MiniMaxH3ReferenceToVideo` observably does, and the difference is not
academic: `wired_labels` reduces soundtracks to a COUNT and pairs
`k < n_vaud`, so it cannot represent a sparse socket set and gets one wrong.
Core, driven with `ref_video_0`, `ref_video_1` and only
`ref_video_audio_1`, emits `video, audio, video` -- the track belongs to the
second clip, by suffix. `wired_labels` puts it on the first. The AGREE cases
below therefore drive core and compare **ordered records**, not label sets;
`wired_labels` gets one case of its own, asserting the disagreement so that a
future fix to it is noticed rather than silently diverging further.

**Not a drift gate.** The repo's no-new-check rule governs instruments added
to catch drift where existing gates already look; this is the acceptance suite
for a component that has none, and it retires with `wired_labels` when the
last socket graph is repointed.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from check_ref_prompt_labels import wired_labels  # noqa: E402
# reference_order is imported per-case, so a case naming a symbol that moved
# fails as that case rather than as a module-level ImportError nobody reads.

MAX_SOCKETS = 3

failures: list[str] = []


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        failures.append(name)
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        failures.append(name)
        print(f"  ERROR {name}: {type(e).__name__}: {e}")
    else:
        print(f"  ok    {name}")


def _core():
    import importlib
    sys.path.insert(0, str(Path.home() / "ComfyUI"))
    # This acceptance suite executes only stubs. Comfy's import path otherwise
    # selects CUDA before the first stub runs, contradicting this file's own
    # no-CUDA contract and needlessly contending with a live render.
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    return importlib.import_module("comfy_extras.nodes_minimax_h3")


def _drive_core(img_ids, vid_ids, track_ids, aud_ids):
    """Core's own (`ref_items` types, `ref_blocks` kinds) for one configuration."""
    import torch
    import check_reference_contracts as CC
    frames = torch.zeros(8, 64, 64, 3)
    kw = dict(
        ref_images={f"ref_image_{i}": torch.zeros(1, 64, 64, 3) for i in img_ids},
        ref_videos={f"ref_video_{i}": frames for i in vid_ids},
        ref_video_audios={f"ref_video_audio_{i}": CC._audio(1.0)
                          for i in track_ids},
        ref_audios={f"ref_audio_{i}": CC._audio(1.0) for i in aud_ids})
    items, blocks = CC._drive(_core(), **kw)
    return [i["type"] for i in items], [b["kind"] for b in blocks]


def _sockets(img_ids, vid_ids, track_ids, aud_ids):
    ins = {}
    for i in img_ids:
        ins[f"ref_images.ref_image_{i}"] = ["x", 0]
    for i in vid_ids:
        ins[f"ref_videos.ref_video_{i}"] = ["x", 0]
    for i in track_ids:
        ins[f"ref_video_audios.ref_video_audio_{i}"] = ["x", 0]
    for i in aud_ids:
        ins[f"ref_audios.ref_audio_{i}"] = ["x", 0]
    return ins


# Socket configurations, chosen so the sparse and reversed cases are present
# rather than only the dense ones a generator happens to emit.
CONFIGS = [
    ("one image", [0], [], [], []),
    ("two images", [0, 1], [], [], []),
    ("silent video", [], [0], [], []),
    ("sounded video", [], [0], [0], []),
    ("image + sounded video", [0], [0], [0], []),
    ("two videos, first sounded", [], [0, 1], [0], []),
    ("SPARSE: two videos, SECOND sounded", [], [0, 1], [1], []),
    ("two videos, both sounded", [], [0, 1], [0, 1], []),
    ("standalone audio only, with an image", [0], [], [], [0]),
    ("image + sounded video + standalone audio", [0], [0], [0], [0]),
    ("everything, sparse track", [0, 1], [0, 1], [1], [0]),
]


def _chain(kinds, sounded=frozenset()):
    """A synthetic append chain: node k+1 takes `references` from node k."""
    cls = {"image": "MiniMaxH3AppendRefImage",
           "audio": "MiniMaxH3AppendRefAudio",
           "video": "MiniMaxH3AppendRefVideo"}
    # Node "9" is a real loader in the graph, not a dangling id: the
    # provenance walker resolves soundtrack links now, so a stand-in that is
    # not in the graph is a malformed chain rather than a fixture.
    g = {"9": {"class_type": "VHS_LoadVideo", "inputs": {}}}
    for k, kind in enumerate(kinds):
        ins = {} if k == 0 else {"references": [str(k), 0]}
        if kind == "video":
            ins["frames"] = ["9", 0]
            ins["video_info"] = ["9", 3]
            if k in sounded:
                ins["soundtrack"] = ["9", 2]
        g[str(k + 1)] = {"class_type": cls[kind], "inputs": ins}
    return g


def agrees_with_core():
    """The plan reproduces BOTH of core's lists, which differ in length.

    Contract 1 is that `ref_items` and `ref_blocks` are not index-aligned: a
    sounded video is two presentation entries and one DiT block. Asserting
    only the first would pass a plan that labelled correctly and paid the
    wrong DiT cost, so both are compared against their own core list.
    """
    from reference_order import legacy_plan, plan_blocks, plan_kinds
    for label, imgs, vids, tracks, auds in CONFIGS:
        plan = legacy_plan(_sockets(imgs, vids, tracks, auds))
        items, blocks = _drive_core(imgs, vids, tracks, auds)
        assert plan_kinds(plan) == items, (
            f"{label} items: plan {plan_kinds(plan)} vs core {items}")
        assert plan_blocks(plan) == blocks, (
            f"{label} blocks: plan {plan_blocks(plan)} vs core {blocks}")
        if any(tracks):
            assert len(items) != len(blocks) or not vids, (
                f"{label}: a sounded video must make the two lists differ")
    print(f"        {len(CONFIGS)} configurations driven through core, "
          f"items AND blocks identical")


def wired_labels_now_shares_the_function():
    """`wired_labels` is a front end over this module, not a second answer.

    **This case replaced a temporary one in the same slice that earned the
    replacement**, which is the difference between it and the recorded core
    gaps elsewhere in this repo. Those describe upstream code nobody here
    owns, so recording them green-while-broken is the honest state. This was
    local code being fixed, so a case asserting it stayed broken would have
    been green-while-broken by choice -- the state a retirement exists to
    prevent.

    What it asserted before the fix: `wired_labels` mispaired a sparse socket
    set, putting a lone `ref_video_audio_1` on the first clip where core puts
    it on the second. It now routes through `legacy_plan`, which pairs by
    suffix, so the disagreement is gone by construction rather than by
    coincidence -- and this case fails if a second implementation ever grows
    back.
    """
    from reference_order import assign_labels, legacy_plan
    for label, imgs, vids, tracks, auds in CONFIGS:
        ins = _sockets(imgs, vids, tracks, auds)
        assert wired_labels(ins) == assign_labels(legacy_plan(ins)), (
            f"{label}: wired_labels has diverged from the shared function")


def chain_traversal_and_reversal():
    """A chain is discovered tail-first and must come back in user order."""
    from reference_order import resolve_chain
    g = _chain(["image", "audio", "video"])
    got = [type(r).__name__ for r in resolve_chain(g, "3")]
    assert got == ["ImageRef", "AudioRef", "VideoRef"], got
    # and the reversal is load-bearing: build the same nodes in the other
    # direction and the plan must follow the LINKS, not the node ids.
    g2 = _chain(["video", "audio", "image"])
    got2 = [type(r).__name__ for r in resolve_chain(g2, "3")]
    assert got2 == ["VideoRef", "AudioRef", "ImageRef"], got2


def interleaved_records_keep_their_places():
    """Image, audio, video, audio, video -- ordinals follow list position."""
    from reference_order import assign_labels, resolve_chain
    g = _chain(["image", "audio", "video", "audio", "video"])
    got = assign_labels(resolve_chain(g, "5"))
    assert got == ["<Picture 1>", "<Audio 1>", "<Video 1>",
                   "<Audio 2>", "<Video 2>"], got


def a_soundtrack_is_never_a_standalone_record():
    """A sounded video is two presentation items and ONE DiT record."""
    from reference_order import VideoRef, plan_kinds, resolve_chain
    g = _chain(["video"], sounded={0})
    recs = resolve_chain(g, "1")
    assert len(recs) == 1 and isinstance(recs[0], VideoRef), recs
    assert recs[0].has_soundtrack, "the soundtrack was not owned"
    assert plan_kinds(recs) == ["audio", "video"], plan_kinds(recs)


def both_models_wired_raises():
    """A chain AND sockets on one node is a partial plan either way.

    Until 2026-08-22 the chain silently won and the sockets vanished. Found by
    self-audit after the fail-loudly contract was flagged as the thing to
    probe, which is where the earlier drafts of this suite kept springing
    leaks: every one has been an assertion about one side of a two-sided
    thing.
    """
    from reference_order import ChainError, plan_for
    g = _chain(["image"])
    ins = {"references": ["1", 0], "ref_images.ref_image_0": ["x", 0],
           "ref_videos.ref_video_0": ["x", 0]}
    try:
        plan_for(ins, g)
    except ChainError as e:
        assert "silently dropped" in str(e), e
    else:
        raise AssertionError("a node wiring both models produced a partial plan")

    # And the ORDER of the two refusals is asserted, not incidental. A node
    # with a malformed chain and sockets has two models wired; that is the
    # root fault, and reporting the link shape first would send a reader to
    # fix the link and straight back here.
    for bad in ([], "1", ["1", 1]):
        try:
            plan_for({"references": bad, "ref_images.ref_image_0": ["x", 0]}, g)
        except ChainError as e:
            assert "silently dropped" in str(e), (
                f"a malformed chain plus sockets reported the link shape "
                f"({str(e)[:60]}...) where two wired models is the root fault")
            continue
        raise AssertionError(f"references={bad!r} plus sockets did not raise")


def each_model_alone_still_resolves():
    """The refusal above must not fire on either model used on its own."""
    from reference_order import plan_for
    g = _chain(["image"])
    assert len(plan_for({"references": ["1", 0]}, g)) == 1
    assert len(plan_for({"ref_images.ref_image_0": ["x", 0]})) == 1
    # and a graph passed alongside socket wiring is not a chain
    assert len(plan_for({"ref_images.ref_image_0": ["x", 0]}, g)) == 1


def degenerate_chain_values_raise():
    """Every value a declared `references` can carry that is not a chain.

    **Key presence is the discriminator, not link truthiness.** A node that
    declares `references` IS an ordered node; there is nothing to fall back
    to, so an empty list, a missing graph, a malformed link or a non-zero
    output slot are all "no plan", never "the legacy plan" and never `[]`.
    All five were accepted before 2026-08-22, four returning an empty list and
    one silently following the wrong slot. Found by codex probing the
    fail-loudly contract, which is the third time that probe has found
    something -- each one an assertion about only one side of a two-sided
    thing.
    """
    from reference_order import ChainError, plan_for, resolve_chain
    g = _chain(["image"])
    cases = [
        ("empty link, graph present", lambda: plan_for({"references": []}, g)),
        ("link but no graph", lambda: plan_for({"references": ["1", 0]})),
        ("link is not a pair", lambda: plan_for({"references": "1"}, g)),
        ("terminal takes slot 1", lambda: plan_for({"references": ["1", 1]}, g)),
    ]
    for label, fn in cases:
        try:
            got = fn()
        except ChainError:
            continue
        raise AssertionError(f"{label}: returned {got!r} instead of raising")

    mid = _chain(["image", "image"])
    mid["2"]["inputs"]["references"] = ["1", 7]
    try:
        resolve_chain(mid, "2")
    except ChainError:
        return
    raise AssertionError("a mid-chain link from the wrong slot was followed")


def a_video_needs_its_metadata():
    """`video_info` is required, because ownership is the record's claim."""
    from reference_order import ChainError, resolve_chain
    g = _chain(["video"])
    del g["1"]["inputs"]["video_info"]
    try:
        resolve_chain(g, "1")
    except ChainError as e:
        assert "video_info" in str(e), e
        return
    raise AssertionError("a video with no metadata resolved, so loaded_fps "
                         "would have nowhere to come from")


def video_links_must_name_their_own_slots():
    """Right shape, wrong output, is the wrong value -- for frames and metadata.

    **The soundtrack is deliberately NOT in this list**, and that is the fix
    for the compatibility blocker rather than an omission. Frames and
    `video_info` name a loader's own outputs because `loaded_fps` is derived
    from them; a soundtrack arrives through whatever processing the graph
    wires, and every sounded graph here trims it, so it is slot-0 of a
    `TrimAudioDuration`. Constraining it to slot 2 rejected the shipped
    population. Provenance is traced instead --
    `the_shipped_soundtrack_wiring_resolves` covers it.
    """
    from reference_order import ChainError, resolve_chain
    for field, bad in (("frames", 1), ("video_info", 0)):
        g = _chain(["video"])
        g["1"]["inputs"][field] = ["9", bad]
        try:
            resolve_chain(g, "1")
        except ChainError as e:
            assert "slot" in str(e), e
            continue
        raise AssertionError(f"{field} from slot {bad} was accepted")


def slots_must_be_real_integers():
    """`int(slot)` swallowed 0.9, "0" and True; every one is a wrong value.

    The same defect as accepting the right shape from the wrong output, one
    level down. `bool` is excluded explicitly: it subclasses `int` and
    `True == 1` would pass a slot check by accident. Found by codex probing
    the validation rather than the shape.
    """
    from reference_order import ChainError, resolve_chain
    for bad in (0.9, "0", True, 2.9, None, [0]):
        g = _chain(["video"])
        g["1"]["inputs"]["frames"] = ["9", bad]
        try:
            resolve_chain(g, "1")
        except ChainError:
            continue
        raise AssertionError(f"frames slot {bad!r} was accepted")


def the_shipped_soundtrack_wiring_resolves():
    """A trimmed soundtrack is the shape this repo actually ships.

    **The over-constrained rule rejected every sounded graph here**, and it
    was written against the shape the rule's author imagined rather than the
    one in `workflows/`. Every sounded graph routes VHS_LoadVideo's audio
    through `TrimAudioDuration` -- the reference pipeline caps a soundtrack at
    the generated duration and ComfyUI does not -- so the record sees
    `[trim, 0]`, never `[loader, 2]`. Caught by codex checking the rule
    against `workflows/build_workflows.py` instead of against its docstring.

    Frames and `video_info` stay strict, because `loaded_fps` comes from the
    metadata. A soundtrack is EXPECTED to be processed, so it is traced back
    instead, and refused only when it reaches a different known loader.
    """
    from reference_order import ChainError, resolve_chain
    base = {"28": {"class_type": "VHS_LoadVideo", "inputs": {}},
            "35": {"class_type": "TrimAudioDuration",
                   "inputs": {"audio": ["28", 2]}}}

    def chain(track, extra=None):
        g = dict(base)
        g.update(extra or {})
        g["1"] = {"class_type": "MiniMaxH3AppendRefVideo",
                  "inputs": {"frames": ["28", 0], "video_info": ["28", 3],
                             "soundtrack": track}}
        return g

    from reference_order import VideoRef, plan_blocks, plan_kinds
    recs = resolve_chain(chain(["35", 0]), "1")
    assert len(recs) == 1 and isinstance(recs[0], VideoRef), recs
    shipped = recs[0]
    assert shipped.has_soundtrack, shipped
    assert shipped.soundtrack_origin == "owned", shipped
    # The two halves of contract 1, on the shape that actually ships: two
    # presentation entries, ONE DiT block, tagged video_audio. Asserting the
    # record resolved without asserting what it costs would leave the
    # expensive half unchecked on the only wiring anyone runs.
    assert plan_kinds(recs) == ["audio", "video"], plan_kinds(recs)
    assert plan_blocks(recs) == ["video_audio"], plan_blocks(recs)
    raw = resolve_chain(chain(["28", 2]), "1")[0]
    assert raw.soundtrack_origin == "owned", raw

    # An unfamiliar audio node is UNRESOLVED, not wrong: it may read the same
    # loader through an input this module does not know.
    unknown = resolve_chain(
        chain(["77", 0], {"77": {"class_type": "Whatever", "inputs": {}}}), "1")[0]
    assert unknown.soundtrack_origin == "unresolved", unknown

    # A different KNOWN source is reported, not refused. Provenance is
    # diagnostic: ownership is established by placing the track in this
    # record, and a deliberate cross-source pairing is legitimate.
    foreign = resolve_chain(
        chain(["99", 2],
              {"99": {"class_type": "VHS_LoadVideo", "inputs": {}}}), "1")[0]
    assert foreign.soundtrack_origin == "foreign", foreign
    assert foreign.has_soundtrack, foreign


def soundtrack_provenance_carries_slots():
    """The walk tracks (node, slot) pairs, because a class name is not a value.

    Every case here reported "owned" against 773dd4f, where the walker
    followed node ids and discarded slots. Found by codex probing the walker
    rather than its inputs. A `VHS_LoadVideo`'s slot 0 is IMAGE, not audio,
    and calling it an owned soundtrack is the class of error the whole
    ownership rule exists to prevent.

    The verdicts split three ways on purpose. A malformed graph -- missing
    node, known class at an output that carries no audio, known processor with
    a broken input, a cycle -- RAISES, because those are "this cannot be
    right". `unresolved` is reserved for an unfamiliar node at a plausible
    output, which may well read the correct loader through an input this
    module does not know.
    """
    from reference_order import ChainError, resolve_chain
    base = {"28": {"class_type": "VHS_LoadVideo", "inputs": {}},
            "35": {"class_type": "TrimAudioDuration",
                   "inputs": {"audio": ["28", 2]}}}

    def chain(track, extra=None):
        g = dict(base)
        g.update(extra or {})
        g["1"] = {"class_type": "MiniMaxH3AppendRefVideo",
                  "inputs": {"frames": ["28", 0], "video_info": ["28", 3],
                             "soundtrack": track}}
        return g

    must_raise = [
        ("loader IMAGE output as a soundtrack", ["28", 0], None),
        ("fractional slot", ["28", 2.9], None),
        ("negative slot", ["28", -1], None),
        ("pass-through output that carries no audio", ["35", 7], None),
        ("trim reading the loader's images", ["35", 0],
         {"35": {"class_type": "TrimAudioDuration",
                 "inputs": {"audio": ["28", 0]}}}),
        ("a node that is not in the graph", ["404", 0], None),
        ("known pass-through with a broken input", ["35", 0],
         {"35": {"class_type": "TrimAudioDuration",
                 "inputs": {"audio": "nope"}}}),
        ("a cycle among pass-throughs", ["35", 0],
         {"35": {"class_type": "TrimAudioDuration",
                 "inputs": {"audio": ["36", 0]}},
          "36": {"class_type": "TrimAudioDuration",
                 "inputs": {"audio": ["35", 0]}}}),
    ]
    for label, track, extra in must_raise:
        try:
            got = resolve_chain(chain(track, extra), "1")[0]
        except ChainError:
            continue
        raise AssertionError(
            f"{label}: resolved to {got.soundtrack_origin!r} instead of raising")

    # SplitAudioChannels carries audio on BOTH outputs, so both resolve.
    for slot in (0, 1):
        g = chain(["44", slot],
                  {"44": {"class_type": "SplitAudioChannels",
                          "inputs": {"audio": ["28", 2]}}})
        assert resolve_chain(g, "1")[0].soundtrack_origin == "owned", slot


def a_cycle_raises():
    from reference_order import ChainError, resolve_chain
    g = _chain(["image", "image"])
    g["1"]["inputs"]["references"] = ["2", 0]          # close the loop
    try:
        resolve_chain(g, "2")
    except ChainError as e:
        assert "cycle" in str(e), e
        return
    raise AssertionError("a cyclic chain resolved instead of raising")


def a_non_builder_link_raises():
    from reference_order import ChainError, resolve_chain
    g = _chain(["image"])
    g["0"] = {"class_type": "LoadImage", "inputs": {}}
    g["1"]["inputs"]["references"] = ["0", 0]
    try:
        resolve_chain(g, "1")
    except ChainError as e:
        assert "LoadImage" in str(e), e
        return
    raise AssertionError("a non-append node was accepted into a chain")


def split_video_ownership_raises():
    from reference_order import ChainError, resolve_chain
    g = _chain(["video"])
    g["1"]["inputs"]["video_info"] = ["77", 3]         # a different loader
    try:
        resolve_chain(g, "1")
    except ChainError as e:
        assert "different decodes" in str(e), e
        return
    raise AssertionError("frames and metadata from different loaders were "
                         "accepted, so loaded_fps would describe another clip")


def video_source_class_is_admitted_explicitly():
    """Slots 0/3 only mean frames/info on a recognized VHS video loader."""
    from reference_order import ChainError, VIDEO_SOURCE_CLASSES, resolve_chain
    for cls in VIDEO_SOURCE_CLASSES:
        g = _chain(["video"])
        g["9"]["class_type"] = cls
        recs = resolve_chain(g, "1")
        assert len(recs) == 1, (cls, recs)

    g = _chain(["video"])
    g["9"]["class_type"] = "UnrelatedFourOutputNode"
    try:
        resolve_chain(g, "1")
    except ChainError as e:
        assert "matching slots 0/3" in str(e), e
        return
    raise AssertionError("an unrelated node at matching slots 0/3 was accepted")


def join_audio_channels_traces_both_branches():
    """One joined output has two provenance branches, neither disposable."""
    from reference_order import ChainError, resolve_chain

    def graph(left, right, extra=None):
        g = _chain(["video"])
        g.update(extra or {})
        g["44"] = {"class_type": "JoinAudioChannels",
                   "inputs": {"audio_left": left, "audio_right": right}}
        g["1"]["inputs"]["soundtrack"] = ["44", 0]
        return g

    same = resolve_chain(graph(["9", 2], ["9", 2]), "1")[0]
    assert same.soundtrack_origin == "owned", same

    foreign_source = {"99": {"class_type": "LoadAudio", "inputs": {}}}
    mixed = resolve_chain(
        graph(["9", 2], ["99", 0], foreign_source), "1")[0]
    assert mixed.soundtrack_origin == "foreign", mixed

    unknown = {"77": {"class_type": "Whatever", "inputs": {}}}
    unresolved = resolve_chain(
        graph(["9", 2], ["77", 0], unknown), "1")[0]
    assert unresolved.soundtrack_origin == "unresolved", unresolved

    malformed = graph(["9", 2], "not-a-link")
    try:
        resolve_chain(malformed, "1")
    except ChainError as e:
        assert "audio_right" in str(e), e
        return
    raise AssertionError("a JoinAudioChannels with a broken branch resolved")


def ownership_differs_from_pairing():
    """A soundtrack on the SECOND of two videos, said directly rather than by suffix."""
    from reference_order import VideoRef, assign_labels, legacy_plan
    by_suffix = assign_labels(legacy_plan(_sockets([], [0, 1], [1], [])))
    owned = assign_labels([VideoRef(), VideoRef(has_soundtrack=True)])
    assert by_suffix == owned, (
        f"the ordered model must be able to SAY what suffixes imply: "
        f"{owned} vs {by_suffix}")
    dense = assign_labels(legacy_plan(_sockets([], [0, 1], [0], [])))
    assert dense != owned, "ownership gained nothing over suffix pairing"


def position_differs_from_trailing_audio():
    """Standalone audio BEFORE a video -- the socket model cannot say it."""
    from reference_order import AudioRef, VideoRef, assign_labels, legacy_plan
    sockets = assign_labels(legacy_plan(_sockets([], [0], [], [0])))
    ordered = assign_labels([AudioRef(), VideoRef()])
    assert sockets == ["<Video 1>", "<Audio 1>"], sockets
    assert ordered == ["<Audio 1>", "<Video 1>"], ordered
    assert sockets != ordered, "ordering gained nothing over fixed iteration"


def shared_audio_counter_is_preserved():
    """One counter across soundtracks and standalone, in emission order."""
    from reference_order import AudioRef, VideoRef, assign_labels
    got = assign_labels([VideoRef(has_soundtrack=True), AudioRef(),
                         VideoRef(has_soundtrack=True)])
    assert got == ["<Audio 1>", "<Video 1>", "<Audio 2>",
                   "<Audio 3>", "<Video 2>"], got


def a_bad_record_raises():
    """An unknown record type must not be silently skipped."""
    from reference_order import ImageRef, assign_labels
    try:
        assign_labels([ImageRef(), "not a record"])
    except TypeError:
        return
    raise AssertionError("a non-record was accepted and produced no label")


def main() -> int:
    print("ordered-reference plan vs core's own reference bookkeeping\n")
    print("  AGREE -- against CORE, comparing ordered records")
    check("the plan reproduces core's ref_items sequence", agrees_with_core)
    check("the shared <Audio j> counter is preserved",
          shared_audio_counter_is_preserved)
    check("wired_labels is a front end, not a second answer",
          wired_labels_now_shares_the_function)
    print("\n  the typed chain")
    check("traversal is tail-first and comes back in user order",
          chain_traversal_and_reversal)
    check("interleaved records keep their list positions",
          interleaved_records_keep_their_places)
    check("a soundtrack is owned, never a standalone record",
          a_soundtrack_is_never_a_standalone_record)
    print("\n  and it fails loudly rather than returning a partial plan")
    check("wiring both models at once raises", both_models_wired_raises)
    check("but either model alone still resolves",
          each_model_alone_still_resolves)
    check("every degenerate `references` value raises",
          degenerate_chain_values_raise)
    check("a video without its metadata raises", a_video_needs_its_metadata)
    check("video links must name their own output slots",
          video_links_must_name_their_own_slots)
    check("output slots must be real integers", slots_must_be_real_integers)
    check("the shipped trimmed-soundtrack wiring resolves",
          the_shipped_soundtrack_wiring_resolves)
    check("soundtrack provenance carries slots, not just node ids",
          soundtrack_provenance_carries_slots)
    check("a cycle raises", a_cycle_raises)
    check("a link to a non-append node raises", a_non_builder_link_raises)
    check("frames and video_info from different loaders raises",
          split_video_ownership_raises)
    check("frames and video_info come from a recognized VHS loader",
          video_source_class_is_admitted_explicitly)
    check("JoinAudioChannels provenance follows both inputs",
          join_audio_channels_traces_both_branches)
    print("\n  DIFFER -- the two behaviours replaced on purpose")
    check("ownership can SAY what suffix pairing only implies",
          ownership_differs_from_pairing)
    check("list position beats fixed iteration: audio before a video",
          position_differs_from_trailing_audio)
    print("\n  and the resolver refuses what it cannot label")
    check("an unknown record type raises", a_bad_record_raises)
    print(f"\n{len(failures)} failure(s)" if failures else
          "\nall ok -- a superset of the socket model, differing only where intended")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
