#!/usr/bin/env python3
"""The longest length anything renders or plans is `h3_rules.MAX_LENGTH`, and no shorter count comes back as a cap.

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
5. **The shipped graphs**: no `length` or `window_frames` input of a graph `graph_paths` walks is over the ceiling;
   a graph that wires one window's `previous` to another (`MiniMaxH3FreezeAudioWindow`) renders at a chained
   length (its `MiniMaxH3Resolution.length`, the frames the latent is made with), and a song node's
   `window_frames` is a chained length (every shipped song graph runs a track longer than one window): the ceiling is not on the audio clock, and the second window would refuse it.
6. **The old count does not come back.** Every tracked `.py` is read for the number, also inside a name (`_pdd_` and
   the digits); each file that holds it is named in `KEPT` with how many times and why (a measured figure, a test's
   input, a statement about the reference pipeline, or a number of something else). A file not named, or one holding
   it more often than named, is red; one holding it less often is red too, so the list stays true.

**What it cannot see.** A cap written another way (the ceiling less a grid step, `max(...)` of the chain used where a
lone window is meant): part 3 reads the builder's windows for every load length, and the planner's where ComfyUI's
core imports; other tools are read only for the digits. Docs, records and `.md` files are not read. Where core does
not import, parts 2 and 3 say what they did not read and stay green on the rest.

No card, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_length_ceiling.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workflows"))
from _lib import bootstrap, case, finish  # noqa: E402

import h3_rules  # noqa: E402

#: Every tracked file that may hold the old count, how many times, and why. Nothing here is a cap on a length.
KEPT = {
    "bench/bench_e2e_h3.py": (1, "a statement of where the reference pipeline (diffusers) stops, not a cap here"),
    "resolution.py": (1, "the node's own tooltip, saying where the reference pipeline stops and that it is no limit on the model"),
    "bench/bench_kitchen_int8_linear.py": (1, "measured: the packed lengths it times were taken at that frame count"),
    "bench/build_mutant_examples.py": (2, "measured: the dated renders it rebuilds were made at that frame count"),
    "bench/check_audio_freeze.py": (48, "a test's input: window arithmetic on the longest chained length, each with its expected output"),
    "bench/check_capture_masked_run.py": (1, "a test's input: a planned window's frame count in a fixture"),
    "bench/check_keyframe_canvas.py": (1, "a test's input: a frame inside a range"),
    "bench/check_prompt_lists.py": (1, "a test's input: a window length in a fixture graph"),
    "bench/check_propose_runs.py": (2, "a pixel coordinate of a box, not a frame count"),
    "bench/check_retraction_consumers.py": (2, "a quoted sentence about a reference length, in a record it checks"),
    "bench/check_sol_observe.py": (1, "a test's input: the length of a fixture render"),
    "bench/check_subject_tracks.py": (8, "a test's input: frame counts of fixture loads"),
    "bench/comfy_capture_nodes/h3_bench_prefix_reuse/__init__.py": (2, "a capture node's fixed input length, as its records were taken"),
    "bench/gen_phaseb_grid.py": (1, "kept byte for byte: dated records name it"),
    "bench/load_chain.py": (1, "measured: the frames of the files of one dated run, in a docstring"),
    "bench/measure_clip_resolution.py": (1, "reasoned: a sampling step worked out for a clip of that length"),
    "bench/measure_clip_tone.py": (1, "reasoned: a sampling step worked out for a clip of that length"),
    "bench/preflight_graph.py": (3, "measured: token counts and an OOM at that frame count"),
    "bench/probe_ref_rows_16604.py": (7, "a probe's inputs: the canvases and length it was run at, and their labels"),
    "bench/run_graph_arms.py": (1, "a dated record of what the arms rendered at"),
    "bench/run_pruning_arms.py": (1, "a dated record of what the arms rendered at"),
    "bench/run_shot_count_ablation.py": (1, "a dated record of what the arms rendered at"),
    "step_x0_observer.py": (2, "measured: a capture's size at that frame count"),
    "workflows/build_workflows.py": (4, "one probe graph's name, id, fixed input length and output prefix (the r2v PDD length probe), as its records were taken"),
    "workflows/h3_config.py": (4, "measured: a render the owner judged and its record's file name, and a token count and an OOM, at that frame count"),
    "bench/check_length_ceiling.py": (0, "this file names the number only as a pattern built from its digits"),
}
NUMBER = re.compile(r"(?<!\d)" + "3" + "45" + r"(?!\d)")    # built from its digits, so this file holds none


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
    over, n, chained, songs = [], 0, 0, 0
    for path in h3_config.graph_paths(REPO / "workflows", include_bench=True):
        n += 1
        g = json.loads(Path(path).read_text())
        nodes = g.values() if isinstance(g, dict) and all(isinstance(v, dict) and "class_type" in v for v in g.values()) else []
        nodes = list(nodes)
        seam = any(x["class_type"] == "MiniMaxH3FreezeAudioWindow" and "previous" in x.get("inputs", {}) for x in nodes)
        chained += seam
        for node in nodes:
            for k in ("length", "window_frames"):
                v = node.get("inputs", {}).get(k)
                song = node["class_type"] == "MiniMaxH3AudioFreezeSong" and k == "window_frames"
                songs += song
                off = (seam and k == "length" and node["class_type"] == "MiniMaxH3Resolution") or song
                if isinstance(v, int) and (v > h3_rules.MAX_LENGTH or (off and v not in chain)):
                    over.append(f"{Path(path).name}: {node['class_type']}.{k} = {v}" + (" (chained, off the chain)" if off else ""))
    assert not over, over
    return (f"{n} graphs walked: no length or window_frames over {h3_rules.MAX_LENGTH}; "
            f"{chained} with a window chained on another and {songs} song node(s), each at a chained length")


def the_number_does_not_come_back() -> str:
    files = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    found = {}
    for f in files:
        p = REPO / f
        if p.is_file():
            k = len(NUMBER.findall(p.read_text(encoding="utf-8", errors="replace")))
            if k:
                found[f] = k
    unnamed = {f: k for f, k in found.items() if f not in KEPT}
    more = {f: (k, KEPT[f][0]) for f, k in found.items() if f in KEPT and k > KEPT[f][0]}
    stale = {f: (found.get(f, 0), c) for f, (c, _why) in KEPT.items() if found.get(f, 0) < c}
    assert not unnamed and not more, f"the number is back: in files not named {unnamed}; more often than named {more}"
    assert not stale, f"KEPT names more than is there (make the list true): {stale}"
    return f"{sum(found.values())} in {len(found)} files, each named with its reason; none a cap"


case("one ceiling", one_ceiling)
case("the chained lengths are the seam rule's", the_chain_is_the_rule)
case("a load up to the ceiling is one window, every frame kept", one_window_every_frame_kept)
case("the song node's window input", the_song_nodes_input)
case("the shipped graphs", the_shipped_graphs)
case("the number does not come back", the_number_does_not_come_back)

if __name__ == "__main__":
    sys.exit(finish())
