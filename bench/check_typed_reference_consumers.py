#!/usr/bin/env python3
"""Static consumers understand the typed ordered-reference chain.

Pure graph check: no ComfyUI import, server, media read, model, or CUDA.  The
resolver remains the ordering authority; this verifies label discovery and
preflight's media adapter consume that authority instead of silently treating
the new conditioner as a reference-free base graph.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from check_ref_prompt_labels import wired_labels  # noqa: E402
from preflight_graph import _reference_media, guide_for  # noqa: E402
from reference_order import ChainError  # noqa: E402


def _graph():
    return {
        "9": {"class_type": "VHS_LoadVideo", "inputs": {"video": "clip.mp4"}},
        "10": {"class_type": "LoadImage", "inputs": {"image": "still.png"}},
        "11": {"class_type": "LoadAudio", "inputs": {"audio": "voice.wav"}},
        "12": {"class_type": "TrimAudioDuration",
               "inputs": {"audio": ["9", 2], "duration": 5.17}},
        "1": {"class_type": "MiniMaxH3AppendRefImage",
              "inputs": {"image": ["10", 0], "size_policy": "max"}},
        "2": {"class_type": "MiniMaxH3AppendRefAudio",
              "inputs": {"references": ["1", 0], "audio": ["11", 0]}},
        "3": {"class_type": "MiniMaxH3AppendRefVideo",
              "inputs": {"references": ["2", 0], "frames": ["9", 0],
                         "video_info": ["9", 3], "soundtrack": ["12", 0]}},
        "4": {"class_type": "MiniMaxH3ReferenceConditioning",
              "inputs": {"references": ["3", 0], "prompt": "typed refs",
                         "width": 1344, "height": 768, "length": 124}},
    }


def main():
    graph = _graph()
    inputs = graph["4"]["inputs"]
    failures = []

    def check(name, fn):
        try:
            fn()
        except Exception as exc:
            failures.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {name}")

    def labels_follow_chain_order():
        got = wired_labels(inputs, graph)
        assert got == ["<Picture 1>", "<Audio 1>", "<Audio 2>", "<Video 1>"], got

    def preflight_selects_reference_guide():
        assert guide_for(inputs) == "ref"

    def preflight_sees_every_owned_medium():
        media, policies, typed = _reference_media(inputs, graph)
        assert typed
        assert media == {
            "ref_images.ref_image_0": ["10", 0],
            "ref_audios.ref_audio_0": ["11", 0],
            "ref_videos.ref_video_0": ["9", 0],
            "ref_video_audios.ref_video_audio_0": ["12", 0],
        }, media
        # `linked` names any of the three that is wired to an input socket
        # rather than carrying a value; empty is the ordinary case.
        #
        # `absent` names any that is on NEITHER the flat nor the dotted
        # spelling. Added to `preflight_graph._reference_media` in `d7dd575`
        # (2026-08-27) so a value it cannot read is recorded rather than
        # silently defaulted -- a silent default is what hid the DynamicCombo
        # rename for a day. This fixture's append node carries `size_policy`
        # alone, so all three are legitimately absent and the values beside
        # them are the defaults, not readings.
        assert policies == {"ref_images.ref_image_0": {
            "size_policy": "max", "allow_upscale": False, "short_edge": 2048, "qwen_short_edge": 0,
            "linked": [],
            "absent": ["allow_upscale", "short_edge", "qwen_short_edge"],
        }}, policies

    def malformed_chain_is_not_partially_reported():
        broken = _graph()
        broken["3"]["inputs"]["video_info"] = ["10", 3]
        for consumer in (
            lambda: wired_labels(broken["4"]["inputs"], broken),
            lambda: _reference_media(broken["4"]["inputs"], broken),
        ):
            try:
                consumer()
            except ChainError:
                pass
            else:
                raise AssertionError("consumer accepted a partial typed chain")

    print("typed reference static consumers\n")
    check("label discovery follows ordered chain", labels_follow_chain_order)
    check("preflight selects ref-en for typed conditioning",
          preflight_selects_reference_guide)
    check("preflight sees every owned medium",
          preflight_sees_every_owned_medium)
    check("malformed chains are never partially reported",
          malformed_chain_is_not_partially_reported)
    print(f"\n{len(failures)} failure(s)" if failures else
          "\nall ok -- typed chains are visible to both static consumers")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
