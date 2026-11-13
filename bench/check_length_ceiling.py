#!/usr/bin/env python3
"""The longest length anything renders or plans is `h3_rules.MAX_LENGTH`, and shipped graphs obey H3 frame & clock rules.

The owner's word of 2026-10-15 (AGENTS.md, "Settled about H3"): 362 is the max, and the shorter count that had stood in
for it as a cap is removed and does not come back. This holds:

1. **One ceiling.** `h3_config.LONG_LENGTH` and `h3_config.MAX_LENGTH` are `h3_rules.MAX_LENGTH`, and
   `h3_config.REF_VIDEO_LENGTH` (the reference clip's own length, no cap) is on the video grid.
2. **The chained lengths are the seam rule's, under the ceiling.** `h3_rules.chain_lengths()` is every count from
   `SHORTEST_CHAINED` up to the ceiling that is on the video grid (17n+5) and whole on the audio clock; `loop_plan`
   and `masked_job` take theirs from it; `h3_config.LONG_WINDOW` is its longest. The audio rate the rule uses is
   core's own (`comfy_extras.nodes_minimax_h3.AUDIO_LATENT_FPS`), read where ComfyUI imports.
3. **A load of up to the ceiling is ONE window, every real frame kept.** For every count up to the ceiling the
   builder's window (`masked_job_graph.window_for`) holds every frame of the load, is on the video grid, and is one
   window (`one_window`, and `loop_plan.stands_alone` where ComfyUI imports); past the ceiling it chains at the
   longest chained length. A window shorter than its load, which would leave the load's last frames as the
   original, is red.
4. **The song node's window input** is capped at the ceiling and defaults to the longest chained length, both read
   from code (`max=MAX_LENGTH`, `default=max(loop_plan.CHAIN_LENGTHS)`), never typed.
5. **The shipped graphs**: across every graph `graph_paths` walks, every literal generation frame count (`length`,
   `window_frames`, `frame_count`) on an H3 latent/conditioning/resolution/song node obeys the `17n + 5` video VAE
   grid up to `h3_rules.MAX_LENGTH` (allowing `length = 1` only on single-frame image graphs); chained
   `MiniMaxH3Resolution.length` (when `MiniMaxH3FreezeAudioWindow.previous` is wired) and
   `MiniMaxH3AudioFreezeSong.window_frames` stay in `chain_lengths()`, every `context_frames` on
   `MiniMaxH3FreezeAudioWindow` / `MiniMaxH3AudioFreezeSong` sits on both clocks (`>= 39` and
   `on_both_clocks(ctx)`), and `MiniMaxH3SoundCues` never sets `snap_to_3f = False`.

**What it cannot see.** A cap written another way in a tool part 3 does not run (the ceiling less a grid step, `max(...)`
of the chain used where a lone window is meant): part 3 reads the builder's windows for every load length, and the
planner's where ComfyUI's core imports. Docs, records and `.md` files are not read. Where core does not import, parts
2 and 3 say what they did not read and stay green on the rest.

No card, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_length_ceiling.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workflows"))
from _lib import bootstrap, case, finish  # noqa: E402

import h3_rules  # noqa: E402

H3_LENGTH_NODES = {
    "MiniMaxH3Resolution",
    "EmptyMiniMaxH3LatentAV",
    "MiniMaxH3Conditioning",
    "MiniMaxH3ReferenceConditioning",
    "MiniMaxH3EncodeReferences",
    "MiniMaxH3ImageToVideo",
    "MiniMaxH3ReferenceToVideo",
    "MiniMaxH3AudioFreezeSong",
}
SINGLE_FRAME_IMAGE_NODES = {
    "MiniMaxH3ImageConditioning",
    "PH3ImageConditioning",
    "MiniMaxH3PH3ImageConditioning",
    "MiniMaxH3ImageSize",
    "MiniMaxH3EncodeImageReferences",
}


def one_ceiling() -> str:
    import h3_config
    assert h3_config.LONG_LENGTH == h3_config.MAX_LENGTH == h3_rules.MAX_LENGTH, (h3_config.LONG_LENGTH, h3_config.MAX_LENGTH, h3_rules.MAX_LENGTH)
    ref = h3_config.REF_VIDEO_LENGTH
    assert ref % h3_rules.FRAME_FACTOR == h3_rules.FRAME_REMAINDER and ref <= h3_rules.MAX_LENGTH, ref
    return (f"h3_config.LONG_LENGTH and MAX_LENGTH are h3_rules.MAX_LENGTH ({h3_rules.MAX_LENGTH}); "
            f"REF_VIDEO_LENGTH ({ref}, from the clip's seconds) on the video grid")


def the_chain_is_the_rule() -> str:
    import h3_config
    import masked_job
    fps = h3_rules.FPS
    want = tuple(n for n in range(h3_rules.SHORTEST_CHAINED, h3_rules.MAX_LENGTH + 1)
                 if n % h3_rules.FRAME_FACTOR == h3_rules.FRAME_REMAINDER and (n * h3_rules.AUDIO_LATENT_RATE) % fps == 0)
    assert h3_rules.chain_lengths() == want and max(want) <= h3_rules.MAX_LENGTH, h3_rules.chain_lengths()
    assert masked_job.chain_lengths() == want, masked_job.chain_lengths()
    assert h3_config.LONG_WINDOW == max(want), h3_config.LONG_WINDOW
    said = f"chained lengths {want}: every count on both clocks up to the ceiling; masked_job and LONG_WINDOW take them"
    try:
        bootstrap(cpu=True)
        from comfy_extras.nodes_minimax_h3 import AUDIO_LATENT_FPS, FPS
        import loop_plan
    except Exception as e:  # noqa: BLE001 -- ComfyUI's core is not importable here
        return said + f"; core NOT read ({type(e).__name__}): loop_plan and the audio rate unchecked"
    assert (h3_rules.AUDIO_LATENT_RATE, h3_rules.FPS) == (AUDIO_LATENT_FPS, FPS), ((h3_rules.AUDIO_LATENT_RATE, h3_rules.FPS), (AUDIO_LATENT_FPS, FPS))
    assert loop_plan.CHAIN_LENGTHS == want, loop_plan.CHAIN_LENGTHS
    return said + f"; loop_plan's the same, the audio rate core's own ({AUDIO_LATENT_FPS})"


def one_window_every_frame_kept() -> str:
    import masked_job
    import masked_job_graph as g
    chain, top = masked_job.chain_lengths(), max(masked_job.chain_lengths())
    both = g.both_clocks(chain)
    for n in range(1, top + 1):
        w = g.window_for(n, chain)
        # THE CASE THAT GOES RED when a load's last real frames are dropped to fit a window, or a load under the longest
        # length on both clocks is split into windows, or a window is off either clock
        assert w >= n and w <= top and w in both and g.one_window(n, w, chain), (n, w)
    for n in range(top + 1, h3_rules.MAX_LENGTH + 300):
        assert g.window_for(n, chain) == top, (n, g.window_for(n, chain))
    said = f"every load of 1 to {top} frames is one window on both clocks holding every frame; longer ones chain at {top}"
    try:
        bootstrap(cpu=True)
        import loop_plan
    except Exception as e:  # noqa: BLE001
        return said + f"; loop_plan NOT read ({type(e).__name__})"
    assert all(loop_plan.stands_alone(g.window_for(n, chain), n) or g.window_for(n, chain) in chain for n in range(1, top + 1))
    assert not loop_plan.stands_alone(h3_rules.MAX_LENGTH + h3_rules.FRAME_FACTOR, h3_rules.MAX_LENGTH + 1), "a lone window past the ceiling was taken"
    return said + "; the planner takes each as a window that stands alone"


def the_song_nodes_input() -> str:
    text = (REPO / "audio_freeze_song.py").read_text()
    call = text[text.index('io.Int.Input("window_frames"'):]
    call = call[:call.index("tooltip=")]
    assert "max=MAX_LENGTH" in call and "default=max(loop_plan.CHAIN_LENGTHS)" in call and not re.search(r"(max|default)=\d", call), call
    return "the song node's window_frames: max=MAX_LENGTH, default=max(loop_plan.CHAIN_LENGTHS), no number typed"


def the_shipped_graphs() -> str:
    import h3_config
    chain = h3_rules.chain_lengths()
    bad, n, chained, songs, contexts, cues = [], 0, 0, 0, 0, 0
    for path in h3_config.graph_paths(REPO / "workflows", include_bench=True):
        n += 1
        g = json.loads(Path(path).read_text())
        nodes = list(g.values()) if isinstance(g, dict) and all(isinstance(v, dict) and "class_type" in v for v in g.values()) else []
        seam = any(x["class_type"] == "MiniMaxH3FreezeAudioWindow" and "previous" in x.get("inputs", {}) for x in nodes)
        is_single_frame_graph = Path(path).parent.name == "image" or any(
            x["class_type"] in SINGLE_FRAME_IMAGE_NODES or x.get("inputs", {}).get("single_frame") is True
            for x in nodes
        )
        chained += seam
        for node in nodes:
            ct = node["class_type"]
            inp = node.get("inputs", {})
            if ct in H3_LENGTH_NODES:
                for k in ("length", "window_frames", "frame_count"):
                    v = inp.get(k)
                    if not isinstance(v, int) or isinstance(v, bool):
                        continue
                    song = ct == "MiniMaxH3AudioFreezeSong" and k == "window_frames"
                    songs += song
                    off = (seam and k == "length" and ct == "MiniMaxH3Resolution") or song
                    if v == h3_rules.SINGLE_FRAME:
                        if not is_single_frame_graph:
                            bad.append(f"{Path(path).name}: {ct}.{k} = {v} (single_frame=1 on non-image workflow)")
                    elif v < h3_rules.FRAME_REMAINDER or v > h3_rules.MAX_LENGTH or (v - h3_rules.FRAME_REMAINDER) % h3_rules.FRAME_FACTOR != 0:
                        bad.append(f"{Path(path).name}: {ct}.{k} = {v} (off 17n+5..{h3_rules.MAX_LENGTH} grid)")
                    if off and v not in chain:
                        bad.append(f"{Path(path).name}: {ct}.{k} = {v} (chained, off the chain {chain})")
            if ct in ("MiniMaxH3FreezeAudioWindow", "MiniMaxH3AudioFreezeSong"):
                ctx = inp.get("context_frames")
                if isinstance(ctx, int) and not isinstance(ctx, bool):
                    contexts += 1
                    if ctx < 39 or not h3_rules.on_both_clocks(ctx):
                        bad.append(f"{Path(path).name}: {ct}.context_frames = {ctx} (off 39+51m dual clock)")
            if ct == "MiniMaxH3SoundCues":
                cues += 1
                if inp.get("snap_to_3f") is False:
                    bad.append(f"{Path(path).name}: {ct}.snap_to_3f = False (audio cues must stay on 3f grid)")
    assert not bad, bad
    return (f"{n} graphs walked: all H3 lengths on 17n+5 <= {h3_rules.MAX_LENGTH} (or t=1 single-frame); "
            f"{chained} chained window(s), {songs} song node(s), {contexts} context_frames on both clocks; "
            f"{cues} sound-cue node(s) with snap_to_3f")


case("one ceiling", one_ceiling)
case("the chained lengths are the seam rule's", the_chain_is_the_rule)
case("a load up to the ceiling is one window, every frame kept", one_window_every_frame_kept)
case("the song node's window input", the_song_nodes_input)
case("the shipped graphs", the_shipped_graphs)

if __name__ == "__main__":
    sys.exit(finish())
