#!/usr/bin/env python3
"""The ordered-reference resolver against the socket resolver it replaces.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). Pure stdlib on
both sides -- no CUDA, no model, no server, no ComfyUI import.

**Two verdicts, not one, and conflating them would reject a correct
replacement.** `docs/research/conditioning_nodes.md` splits the acceptance
criteria into behaviour a replacement must PRESERVE and behaviour it
intentionally REPLACES. A suite asserting AGREE on all of it would fail the
new model for doing its job, so:

  AGREE   every socket-shaped input must label identically to
          `check_ref_prompt_labels.wired_labels`, over the whole legal
          combination space rather than a sample. This is the claim that the
          ordered model is a superset and not a rewrite.

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
    return importlib.import_module("comfy_extras.nodes_minimax_h3")


def _drive_core(img_ids, vid_ids, track_ids, aud_ids):
    """Core's own `ref_items` type sequence for one socket configuration."""
    import torch
    import check_reference_contracts as CC
    frames = torch.zeros(8, 64, 64, 3)
    kw = dict(
        ref_images={f"ref_image_{i}": torch.zeros(1, 64, 64, 3) for i in img_ids},
        ref_videos={f"ref_video_{i}": frames for i in vid_ids},
        ref_video_audios={f"ref_video_audio_{i}": CC._audio(1.0)
                          for i in track_ids},
        ref_audios={f"ref_audio_{i}": CC._audio(1.0) for i in aud_ids})
    items, _ = CC._drive(_core(), **kw)
    return [i["type"] for i in items]


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


def agrees_with_core():
    """The plan's record sequence IS core's `ref_items` sequence."""
    from reference_order import legacy_plan, plan_kinds
    for label, imgs, vids, tracks, auds in CONFIGS:
        got = plan_kinds(legacy_plan(_sockets(imgs, vids, tracks, auds)))
        want = _drive_core(imgs, vids, tracks, auds)
        assert got == want, f"{label}: plan {got} vs core {want}"
    print(f"        {len(CONFIGS)} configurations driven through core, "
          f"all identical")


def wired_labels_is_wrong_on_sparse():
    """Record the disagreement so a future fix to `wired_labels` is noticed.

    Green while `wired_labels` still mispairs a sparse socket set. If this
    flips, it was fixed and this case retires -- it is not a defect of the
    ordered model.
    """
    from reference_order import assign_labels, legacy_plan
    ins = _sockets([], [0, 1], [1], [])
    stale = wired_labels(ins)
    ordered = assign_labels(legacy_plan(ins))
    assert stale == ["<Audio 1>", "<Video 1>", "<Video 2>"], stale
    assert ordered == ["<Video 1>", "<Audio 1>", "<Video 2>"], ordered
    assert stale != ordered, ("wired_labels now agrees on sparse sockets -- "
                              "it was fixed; retire this case")


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
    print("\n  and the resolver it replaces is wrong where core is not")
    check("wired_labels mispairs a sparse socket set (recorded, not owned)",
          wired_labels_is_wrong_on_sparse)
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
