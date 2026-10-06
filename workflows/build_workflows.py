#!/usr/bin/env python3
"""Generate the MiniMax H3 test workflows, in API format.

Why a generator instead of hand-edited JSON: the three bundled ComfyUI
templates are not equally editable. `video_minimax_h3_r2v` is a flat graph,
but t2v and i2v hide the entire sampler stack inside a subgraph named
"Image to Video (MiniMax H3)". Editing a subgraph by hand -- or converting
one to API format by hand -- is how you end up measuring a graph that is
subtly not the one you meant to run. Building them all from one description
keeps them identical everywhere they should be identical, and makes the
things that differ (which conditioning node, which checkpoint, whether a LoRA
is applied) obvious.

The sage node goes between `UNETLoader` and the sampler stack. Note that
MODEL forks to *two* consumers -- `BasicScheduler.model` and
`BasicGuider.model`. Rewiring only the guider leaves the scheduler reading
sigmas off the unpatched model; the render still succeeds, which is why the
mistake survives. Every graph here is generated from a single `model_src`
variable so the fork cannot drift.

Run it to regenerate:

    <comfy-venv-python> build_workflows.py

It writes the JSON next to itself and validates every API graph against a
live ComfyUI's /object_info (or a cached copy passed with --object-info).
Validation is static -- nothing is submitted and no model loads. Building does
import ComfyUI core, which opens a CUDA context when a card is visible; while
another process renders, run it as `CUDA_VISIBLE_DEVICES= <comfy-venv-python>
build_workflows.py` and core takes its CPU path (`_core_cpu_when_no_card`,
`docs/checks.md` "While a render is on the card").

**Generation is byte-deterministic, and checking that the shipped graphs are
current costs about twelve seconds and needs no server:**

    build_workflows.py --out "$TMP" --no-validate && diff -rq "$TMP" workflows

Zero differing files means the tree matches this file. Measured 2026-08-31 by
two sessions independently, at 11.4s, 11.6s and 12.1s, no diff each time.
Nobody knew this was cheap, which is why the rule above ("nothing is true of a
graph until it is rebuilt") had been enforced by remembering to run it.

The file COUNT is deliberately not quoted here. The first version said 157,
which is neither the 156 at the top level nor the 159 including `bench/`, and
no reader's next action changes on the number -- the command's own output is
the answer. The rule (`docs/prose_measurements.md`): substitute a different plausible
value, and if nothing changes, the number is decorative.

Deliberately NOT a check in `bench/`. The state it catches -- generator edited,
graphs not rebuilt -- is real, and it happened on 2026-08-31 when this file
emitted `head_strength -1.0` while all 20 shipped graphs still carried 1.0. But
that divergence lived under an hour and was caught by reading `git status`
inside the session that made it, so it is not an escape. And a gate asserting
freshness is CORRECTLY red for most of any session that touches this file,
which is the shape people learn to run last. Freshness matters at COMMIT: if
this is ever wanted as a gate, it belongs in the pre-commit hook, where it
fires once at the moment the answer matters. What would change that: a
divergence that actually ships, i.e. survives the session that made it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# Registry id from pyproject's [tool.comfy], and the nodes this pack owns.
# Kept beside each other so a node added to one and not the other is visible.
_CNR_ID = "comfyui-h3-explorations"
_OUR_NODES = {
    "MiniMaxH3SageAttention",
    "MiniMaxH3Resolution", "MiniMaxH3Preflight",
    "MiniMaxH3ProvenanceStamp", "MiniMaxH3FreezeAudio", "MiniMaxH3FreezeAudioWindow",
    "MiniMaxH3EncodeTrack", "MiniMaxH3AudioAttentionGain",
    "MiniMaxH3AudioFreezeSong", "MiniMaxH3PromptList",
}

# Model names, sampler settings, canvas geometry and the SolAttn knobs all
# used to live here in duplicate with the bench. Single source is
# h3_config.py -- see its docstring for why that matters.
# Every shipped prompt is a prompt_bank/ entry, loaded by id; the constant
# names stay because the bench and the catalogue import them. Adopted
# 2026-09-03 (owner): one source of truth for prompt text.
from prompts import text as _bank_prompt  # noqa: E402
import prompts as _prompts  # noqa: E402
from h3_config import (  # noqa: E402
    DAILY_DIR, DAILY_GRAPHS, SINGLE_FRAME_DIR,
    CORE_LOADED_ENCODERS, IMAGE_VAE, DRAFT_VAE, STEP_SWITCH_PASS1_SIGMAS, STEP_SWITCH_PASS2_SIGMAS,
    STEP_SWITCH_REV, STEP_SWITCH_BASE, CANVAS, FPS, LENGTH, LONG_LENGTH, MODELS,
    SAMPLING, SAGE_NODE, DENSE_BACKEND_NODE, DENSE_CHAINS, DEFAULT_DENSE_CHAIN, SEED, SIGMA_SHIFT, SOL_CORE_NODE, SOL_CORE_DEFAULTS,
    REF_VIDEO_LOADER, REF_QWEN_SHORT_EDGE, SEGMENTER, SUBJECT_TRACK, MASKED_SOURCE,
    MASKED_MOTION_STEPS, MASKED_MOTION_SOURCE, MASKED_PROMPT, MASKED_PROMPT_NODE,
    SAPIENS2, SUBJECT_PARTS, MASKED_PARTS_SOURCE, MASKED_PARTS_PROMPT,
    CACHE_NODE, CACHE_NODE_CLASS,
    DISTILL_SAMPLING,
    REF_VIDEO_BUDGET,
    CAPTURE_REF_IMAGES,
    DIALOGUE_REF_IMAGES, REFVIEW2_SCENES,
    PDD_MANUAL_EVALS,
    PDD_MANUAL_SIGMAS,
    sol_for_graph, SOL_DISTILL_LORA_FILES,
    PDD_FL2VA_LORA, PDD_REF2VA_LORA, PDD_STEPS, PDD_STEPS_FAST,
    PDD_STRENGTH, PDD_FL2VA_STRIPPED_LORA,
    AUDIO_REFINE, FROZEN_VIDEO_CACHE, FROZEN_VIDEO_CACHE_NODE, FLASHGEN_LORA,
    FLASHGEN_R64_LORA, FLASHGEN_R64_REF2VA_LORA, LORA_BRANCH_NODE, FLASHGEN_STRENGTH, FLASHGEN_STEPS,
    FLASHGEN_MANUAL_SIGMAS, FLASHGEN_SAMPLER,
    FASTH3_STEPS, FASTH3_SAMPLER, FASTH3_SCHEDULER, FASTH3_SHIFT, FASTH3_CORE_VSA,
    FASTH3_CONTRACT_SIGMAS, FASTH3_CONTRACT_SAMPLER, FASTH3_CONTRACT_VSA, STEP_SWITCH_FASTH3,
    refine_scheduler_ids,
)


def _composed_from_bank(composed: str) -> str:
    """The bank's copy of a COMPOSED prompt, refusing one that is not in it.

    `_ref_prompt` builds a ref2va prompt from the role tables rather than
    from one literal, so before 2026-09-03 its output was the one class of
    shipped prompt with no home outside the graphs: `prompt_catalogue.md`
    named those scenes `derived:<graph>` because no constant held them and
    nothing could name them. The composition is still what decides the text
    -- the role tables are the reason a prompt declares exactly the labels
    its arm wires -- so this does not replace it. It joins it to the bank:
    compose, look the result up by its own text, and ship the bank's copy.

    So the invariant `prompts.py` states holds for composed prompts too: a
    prompt that is not in the bank cannot ship. A new reference combination
    fails the build here, naming the file to write, rather than emitting a
    graph carrying text no record can identify.
    """
    from prompts import identify  # local: keeps the module import list flat
    pid = identify(composed)
    if pid is None:
        raise SystemExit(
            "a composed prompt is not in the bank. Write it to "
            "prompt_bank/<id>.txt, add its manifest entry (mode ref2va needs "
            "a donor), run bench/build_prompt_bank.py, then rebuild:\n\n"
            + composed)
    return _bank_prompt(pid)


def _retimed_from_bank(prompt_id: str, alignment, length: int) -> str:
    """A keyframe prompt from the bank with its Part One line re-resolved.

    The two keyframe defaults are the one place a shipped prompt is
    genuinely parametric: `base_en.md:14-31` puts the effective duration in
    the alignment sentence to two decimals, so the text a graph carries is a
    function of its frame count. Typing the duration into the bank file and
    stopping there would make the bank right for one length and silently
    wrong for every other -- the defect `fl2v_prompt` was written as a
    function to avoid.

    So the bank holds the whole prompt at the frame count its manifest entry
    declares, which is what `--check` grades and what a reader sees, and this
    swaps the first paragraph for the same sentence resolved at `length`.
    The equality assertion is the join: the template and the bank must agree
    at the declared count, so editing either alone fails the build.
    """
    from prompts import entry as _bank_entry
    text = _bank_prompt(prompt_id)
    head, sep, rest = text.partition("\n\n")
    declared = _bank_entry(prompt_id)["frames"]
    want = alignment(duration_of(snap_length(declared)))
    if head != want:
        raise SystemExit(
            f"prompt_bank/{prompt_id}.txt's Part One line is not the one this "
            f"generator resolves at its declared {declared} frames.\n"
            f"  bank:      {head}\n  generator: {want}")
    return alignment(duration_of(snap_length(length))) + sep + rest


# The Sol-Attn node every graph wires. Fourth node in this slot: kijai's Triton
# pack (`SolAttnPatch`) until 2026-08-14, the vendored upstream CUDA node
# (`SolAttnMiniMax`), ours (`MiniMaxH3SolAttn`) from 2026-08-30, and
# `MiniMaxH3Sol` from the redesign (2026-09-27,
# `docs/research/2026-09-27_sol_node_redesign.md`). A new node id rather than
# an edit, because editor-saved graphs match `widgets_values` by position;
# `MiniMaxH3SolAttn` is deleted once every graph here is regenerated.
SOL_NODE = "MiniMaxH3Sol"

# `token_routing` is a DynamicCombo: its `custom` option brings a `blocks`
# input, which the API form keys under the combo's id with a dot,
# `token_routing.blocks`, and ComfyUI regroups into the dict the node receives
# (`comfy_api/latest/_io.py::build_nested_inputs`).
SOL_ROUTING_NESTED = {"custom": ("blocks",)}
# The node's widgets in declared order (`model` is a socket).
SOL_WIDGETS = ("tau", "quantizer", "dense_blocks", "sink_conditioning", "token_routing",
               "start_percent", "end_percent", "min_tokens", "verbose")


def _distill(lora, pdd, key):
    """Sampler/scheduler default for one graph, from whether it carries a distill.

    Owner decision 2026-08-27: every arm running a distillation LoRA samples on
    `DISTILL_SAMPLING` -- euler/simple -- and everything else keeps `SAMPLING`.
    Derived from what the graph IS rather than retyped per call site, which is
    how the turbo arms ended up split across two samplers while every PDD arm
    passed `sampler_name="euler"` by hand.

    `pdd` alone is not enough: a turbo arm carries `lora` with no `pdd` flag,
    and the split-pack arms carry both. Passing `sampler_name=` at a call site
    still wins, so a deliberate deviation stays possible and stays visible.
    """
    if pdd or lora:
        return DISTILL_SAMPLING["sampler" if key == "sampler" else "scheduler"]
    return SAMPLING[key]


def sol_api_inputs(sol):
    """API-form inputs for the Sol node from a recipe dict (SOL_RECOMMENDED_CUDA
    and its overrides). A custom token-routing list travels as
    `token_routing.blocks`; a key the node does not declare is refused here
    rather than at queue time."""
    nested = SOL_ROUTING_NESTED.get(sol.get("token_routing"), ())
    known = set(SOL_WIDGETS) | {f"token_routing.{k}" for k in nested}
    unknown = sorted(set(sol) - known)
    if unknown:
        raise KeyError(f"Sol config carries {unknown}, which {SOL_NODE} does not declare "
                       f"(token_routing {sol.get('token_routing')!r})")
    return dict(sol)


# Prompts for the long presets (362 frames, 15.083s). A 15s request needs a
# shot timeline, not one continuous beat -- numbered shots (the guide's format
# puts a time on every later shot header; the house writes none, `docs/prompting.md`
# section 3.1), and a 15s request against a 6s prompt leaves the model twelve
# seconds it was never told about.
#
# Laid out per the owner's v6 t2v conditioning format (2026-08-20): each field
# name alone on its line, content on the next, one empty line between fields,
# `N/A` for an empty field.
#
# **Rewritten 2026-08-22, and this is a content change, not a reformat.** The
# previous prompt ran four shots of a cyclist in heavy rain and had NO dialogue.
# Two things were wrong with it:
#
#   No speech. H3 generates dialogue, the guide devotes section 4.4 to asking
#   for it, and nothing shipped here exercised that path. All three below use
#   two speakers: the identifying phrase, the (S1)/(S2) id, the action and the
#   delivery sit OUTSIDE `<d>`, only the language tag and verbatim words sit
#   inside, and each block is followed by the speaker's lips closing.
#
#   Its soundscape asked for continuous texture -- "steady heavy rain on
#   asphalt and metal, tyre hiss through standing water" -- and the 4-step
#   students render that as a TONE. Measured 2026-08-22
#   (`bench/results/2026-08-22_audio_hum.json`): a 400 Hz peak 16-25 dB above
#   its own noise floor under a turbo LoRA, 5.7 dB on base, against 0.3 dB in
#   clips asking for "natural ambient atmosphere". The soundscapes below are
#   EVENT-driven -- crates, coins, a drawer latching -- not continuous hiss.
#
# **The default carries `<d>` and nothing else, deliberately.** The other five
# markers are undocumented in the guide and their encoder rows are UNTRAINED
# (`bench/audit_h3_token_embeddings.py`), so what they do is unmeasured -- and if
# `<|caption_start|>` burns text into the frame, a default carrying one would
# put text in every shipped t2v render. They live in the two opt-in scenes
# instead, where a test can reach them without contaminating the default.
#
# **Changing this makes every prior t2v render a different sample**, which is a
# non-event for comparisons wholly before or wholly after, but means a number
# from before today cannot sit beside one from after without saying so.
#
# Marker coverage across the set is all seven: `<d>`/`</d>` everywhere,
# `<|lyrics_start|>`/`<|lyrics_end|>` in the rehearsal scene wrapping the sung
# `<d>` blocks, `<|caption_start|>`/`<|caption_end|>` and `<|cutoff|>` in the
# clinic scene. Patterns follow `bench/audit_h3_marker_tokenization.py`'s
# scenes, which are the only worked examples of the five the guide omits.
# `t2va_covered_market` is the long preset's scene. An earlier text of it broke
# the guide's camera vocabulary and its identity-at-first-appearance rule and
# graded GREEN, because nothing then checked them; the rules, and which
# readings of the guide are NOT rules, are `docs/prompting.md` sections 4, 5.2
# and 8, and the escape that led to `bench/check_camera_vocabulary.py` is the
# row in `docs/checks.md`. The owner disqualified that text as a SAMPLE on
# 2026-08-27 (`docs/research/pdd/queued_arms.md`): a render of it is evidence
# about that text only, and clips of this scene rendered before 2026-08-28 are
# not matched-seed comparable with later ones. This block used to restate the
# rules item by item and was a second copy of the manual.

LONG_T2V_PROMPT = _bank_prompt("t2va_covered_market")
# The bench pair's scene (see the bench block near the end of main).
BENCH_T2V_PROMPT = _bank_prompt("t2va_frontier_standoff")

# Sung lines: the lyrics markers WRAP one or more `<d>` blocks rather than
# replacing them.
T2V_REHEARSAL_PROMPT = _bank_prompt("t2va_rehearsal_room")

# On-screen text and a line truncated by the end of the clip. `<|cutoff|>` sits
# directly against the closing `</d>` with no full stop between -- a full stop
# before it is dragged into `.<` by BPE on an unpatched tokenizer, so the
# spacing is load-bearing (`docs/comfyui_vendor_gaps.md`).
T2V_CLINIC_PROMPT = _bank_prompt("t2va_clinic_corridor")


#: **The baseline t2v scene set (owner decision, 2026-08-22).** Future tests
#: draw from this rather than from one scene: three settings, three sound
#: worlds, three speaker pairs, so a result that only holds in a market aisle
#: is visible as such. It replaced a single cyclist-in-rain prompt that every
#: t2v measurement in this repo had been taken on.
#:
#: `market` is also the shipped graph default, and is the only one carrying no
#: marker beyond `<d>` -- see the note above the prompts for why the untested
#: five stay out of a default.
#:
#: Ordered, and the order is the sweep order. Keep it stable: a scene set that
#: reorders makes "scene 2" mean different things in two records.
# **The stress scenes, added 2026-08-22.** The three above are each one thing
# done properly; these two are everything at once, which is a different test.
# Fast two-person exchanges overlapping, singing over the top, high motion and
# fast reframing, and burned-in text, all inside 15.083s. Four shots each, so
# the cut rate is roughly one every 3.5 seconds.
#
# The point is that the scenes above cannot fail in an interesting way. A
# single speaker in a quiet room either works or does not; it will not show
# speaker identity bleeding between two voices under motion, or singing
# collapsing into speech, or a caption surviving a whip pan. **These are for
# finding the failure, not for judging quality** -- read them as briefs met.
T2V_SUBWAY_PROMPT = _bank_prompt("t2va_subway_platform")

T2V_KITCHEN_PROMPT = _bank_prompt("t2va_restaurant_kitchen")


T2V_SCENES = {
    "market": LONG_T2V_PROMPT,
    "rehearsal": T2V_REHEARSAL_PROMPT,
    "clinic": T2V_CLINIC_PROMPT,
    "subway": T2V_SUBWAY_PROMPT,
    "kitchen": T2V_KITCHEN_PROMPT,
}


# h264-mp4 rather than h265 or an nvenc variant: software x264 at crf 19 is
# the most portable mp4 there is, and the nvenc paths trade quality per bit
# for encode speed on a file that takes seconds to write next to a render
# that takes minutes. Switch to video/h265-mp4 if size matters more than
# playing everywhere.
VIDEO_FORMAT = "video/h264-mp4"

# Placeholder input filenames. These are whatever the local install happens
# to have; swap them for your own before running an i2v or r2v graph.
# A reference VIDEO is an IMAGE batch, not a VIDEO: `ref_videos.ref_video_0`
# takes frames. A VHS loader (`h3_config.REF_VIDEO_LOADER` says which) because
# VHS's loaders expose `force_rate`, and force_rate=24 is not optional here.
# ComfyUI's node has no fps input at all and assumes 24 twice over -- for the DiT's temporal clock
# and for the `<T.T seconds>` labels the conditioner reads -- while the
# reference pipeline resamples onto 24 from the rate the container reports.
# A 30 fps source left at force_rate=0 is conditioned at the wrong speed,
# silently, and diffusers' own docstring flags exactly this.
# 960x544, 25 fps, 14.4s, WITH an audio track. Three properties earn it: 25 fps
# so force_rate=24 has visible work to do, a soundtrack so the paired <Audio 1>
# path is exercised rather than skipped, and a length that MATCHES the render.
#
# **Trimmed from a 19.56s original on 2026-08-22, and the trim is the point.**
# The model tops out at 362 frames / 15.083s -- `MiniMaxH3Resolution`'s tooltip
# and `h3_rules.MAX_LENGTH` -- so 4.46s of that source, 23% of a continuous
# monologue, was cut wherever 362 frames happened to land. The reference kept
# talking past the end of the render and the last third of every render
# drifted; the owner heard it before any measure showed it, and
# `bench/results/2026-08-22_swap_prompt_verdict_362.json` has the per-third
# numbers. The cut is at 14.375s, inside a 0.3s silence at -56 dB, so the
# utterance ENDS rather than being interrupted -- and 14.375 * 24 is exactly
# 345, the `17n+5` count `REF_VIDEO_LENGTH` renders at.
#
# **Still 25 fps on purpose.** Trimming to 24 would have made `force_rate=24` a
# no-op and quietly retired the fps hazard this clip exists to exercise. The
# problem was the length, so only the length changed.
#
# The 19.56s original stays in the input root, referenced by nothing.
# In the input ROOT, not `h3_refs/`: the VHS loader's `video` widget is a combo
# of root filenames and lists no subfolder paths, so a graph naming one fails
# the served-schema validation. Found by that validation, which is what it is
# for. The `h3ref_` prefix keeps it grouped with the fps probe clips instead.
PLACEHOLDER_VIDEO = "h3ref_diner_monologue_25fps_14s.mp4"
# Kept in the input directory but used by NO shipped graph. They exist to make
# the force_rate hazard reproducible: three 6.00-second clips trimmed to differ
# only in frame rate, so the 0% / +4.2% / +25.0% timeline errors in the note
# below can be re-derived rather than trusted. Built with
#   ffmpeg -ss 2 -t 6 -i <src> -c:v libx264 -crf 18 -c:a aac <dst>
# from LTX-2_00010-audio1.mp4 (24), 20260601_172336_00001-audio.mp4 (25) and
# The_Pavement_Turns_To_Carpet.mp4 (30). Safe to delete; nothing references them.
_FPS_PROBE_CLIPS = ("h3ref_24fps_6s.mp4", "h3ref_25fps_6s.mp4",
                    "h3ref_30fps_6s.mp4")
# Silent, for the video-only arm. **VHS RAISES when its audio output is wired
# on a clip with no audio stream** -- "VHS failed to extract audio from ..." --
# so a video-only graph has to leave that socket unwired rather than lean on
# the downstream node treating it as optional. Found by running it, not by
# reading: the graph validated fine and died at execution.
PLACEHOLDER_VIDEO_SILENT = "LTX-2_00065.mp4"
# Standalone audio reference. The reference refuses one that is not paired
# with at least one image or video, so it never appears alone here.
PLACEHOLDER_AUDIO = "4th-ninja-Breathless_Heights.mp3"
REF_VIDEO_FORCE_RATE = 24.0

# Verified present in ComfyUI's ACTUAL input directory, which on this install
# is not under the ComfyUI tree -- `folder_paths.get_input_directory()` is
# authoritative and a bare `ls ComfyUI/input` is not. Getting that wrong on
# 2026-08-13 produced a "29 of 30 combo entries are stale" conclusion that was
# entirely an artifact of looking in the wrong place.
PLACEHOLDER_IMAGE_A = "1-man.png"
PLACEHOLDER_IMAGE_B = "2-mountain_landscape.png"

# The LoadImage id of each reference slot, in socket order.
#
# **Fixed per slot rather than allocated in a loop.** `bench_e2e_h3.py` addresses
# the first slot as "15" by name, so a renumbering would silently point a bench
# at the wrong node. Slot 3 takes 34 because 26-33 and 40-43 are already spoken
# for in this graph.
#
# Slots 4-6 were added 2026-08-18 for the workload-grid count ladder. Typed
# append ids are allocated separately below, so these remain only the stable
# loader ids benches address. Slots 4-6 take 36, 38 and 45: 26-33 and 40-43 are
# spoken for (reference loaders, split path, plain chain), and 44 is the cache
# node. Each slot also reserved the next id for a `MiniMaxH3ReferenceFit`
# (24, 25, 35, 37, 39, 46), retired in 0.173.0 and no longer emitted.
_REF_IMAGE_NODES = ("15", "16", "34", "36", "38", "45")

# Typed reference append nodes, in presentation order. The six image slots
# above plus one video and one standalone audio reference can consume all
# eight. These ids are outside the long-standing 1-47 API graph allocation so
# the migration does not renumber nodes that benches address directly.
_REF_APPEND_NODES = tuple(str(i) for i in range(50, 58))

# Prompt List ids on the song graphs, in chain order; nothing else here uses 90-97.
_PROMPT_LIST_NODES = tuple(str(i) for i in range(90, 98))

# The lists of the shipped prompt-list example (the owner asked for one,
# 2026-09-14; the values are the session's), as (name, values one per line,
# order, shuffle). The second shot of `prompt_bank/t2va_song_flicker_lists.txt`
# reads "a medium shot of her __place__, __motion__," and each value finishes
# that clause. Different lengths and orders, so the two turn over at
# different sections.
_SONG_FLICKER_LISTS = (
    ("place", "\n".join((
        "standing at the rain-streaked window of a bare concrete apartment",
        "leaning on the rail of a brutalist high-rise balcony above the city",
        "sitting alone by the window of an empty late-night train carriage as city lights slide past",
        "walking slowly down a wet, empty side street under a flickering sodium lamp",
        "sitting on a stairwell landing beneath a flickering fluorescent tube",
    )), "shuffled", 0),
    ("motion", "\n".join((
        "swaying slowly on the laid-back beat, eyes half closed",
        "tilting her head on each soft backbeat, arms folded against the cold",
        "turning slowly away from the camera, then glancing back over her shoulder",
        "pulling her sleeves down over her hands and nodding gently in time",
    )), "in_order", 0),
)

# The example's timeline: the section starts of `just-a-flicker.mp3` from the
# analysis the owner supplied (2026-09-14), so the place changes where the
# song changes section rather than where a window ends.
_SONG_FLICKER_TIMELINE = "\n".join((
    "00:00 intro", "00:10 verse", "00:32 chorus", "00:53 verse",
    "01:14 chorus", "01:35 bridge", "01:57 chorus", "02:18 outro",
))


#: The named attention modes an entry's `dense_attn` may carry. A mode gets a
#: name, never a number (bench/check_literal_widgets.py's rule, applied to the
#: generator's own extras).
# "ck" (2026-09-15): the community chain. ComfyUI core's Model Attention
# Backend node at h3_config.DENSE_BACKEND_NODE (kitchen's int8_attention, which
# rotates q/k before INT8 and is immune to the loud-channel mechanism) as the
# dense kernel, our Sol node on top, sage ABSENT. Added that morning to measure
# the block-49 effect on the chain most people run
# (bench/results/2026-09-15_ck_int8_attention_block49.json); **the default
# since that evening, owner decision**, so an entry that names no mode gets it.
# "sage_sol" (2026-09-15): the chain every video graph shipped until then, sage
# under Sol, for the arms that must stay on it;
# bench/check_attention_defaults.py::FLOOR_STEMS says which and why.
_DENSE_ATTN_MODES = ("none", "sage", "sol", "ck", "sage_sol")


def _sol_with_overrides(extra: dict) -> dict:
    """The Sol config a GRAPHS entry carries: `h3_config.sol_for_graph` for
    its PDD state and step count, with the entry's `sol_overrides` laid over
    it. Overrides exist for the candidate graphs (2026-09-05): a window, a
    sink mode or a tau that the shipped recipe does not carry, baked into the
    widget so the owner can open the graph and render their own prompt on
    it. Every overridden field is a declared deviation in
    `bench/check_attention_defaults.py::DEVIATIONS`, which also asserts the
    deviation is real; an override that matches the recipe is refused here
    so a stale one cannot ride along as a no-op."""
    lora = (extra.get("lora") or (None,))[0]
    base = sol_for_graph(bool(extra.get("pdd", False)),
                         extra.get("steps", SAMPLING["steps"]),
                         distill_lora=lora in SOL_DISTILL_LORA_FILES)
    over = extra.get("sol_overrides") or {}
    unknown = sorted(set(over) - set(base))
    if unknown:
        raise SystemExit(f"sol_overrides names fields the recipe lacks: {unknown}")
    same = sorted(k for k, v in over.items() if base[k] == v)
    if same:
        raise SystemExit(f"sol_overrides restates the recipe's own value for {same}; "
                         "drop it or change it")
    return dict(base, **over)


def _attention_plan(extra: dict) -> tuple[bool, bool, str | None]:
    """(sage, sol_on, dense_mode) from one GRAPHS entry's extras.

    Default, since 2026-09-15 (owner): "ck", core's Model Attention Backend
    at h3_config.DENSE_BACKEND_NODE as the dense kernel and Sol on top, no
    sage. Until then the default was sage AND Sol, which is "sage_sol" now.
    The owner's standing direction (2026-08-17) is unchanged: Sol-Attn is on
    by default on every video workflow; `sol_on=False` bypasses it for a
    named test and leaves the dense kernel alone.

    `dense_attn` names the dense kernel, and it is a named mode rather than a
    flag because there are five:

      "ck"            the default: the kitchen backend node, Sol on unless
                      `sol_on=False`.
      "sage_sol"      sage with Sol on top, the chain every video graph
                      shipped until 2026-09-15, for the arms that must stay
                      on it (bench/check_attention_defaults.py::FLOOR_STEMS).
      True or "none"  neither a dense node nor Sol: whatever kernel ComfyUI
                      resolves on its own. For probes whose subject is a
                      numerical mechanism elsewhere in the model, since every
                      attention node changes numerics; and the PDD reference
                      arms, which replicate the vendor's Diffusers path.
      "sage"          sage with Sol ABSENT. Absent rather than bypassed, for
                      PDD: Sol skips attention adaptively per step, which is
                      incoherent against a fixed fused block schedule, and a
                      bypassed node in the graph is an invitation to switch
                      it on.
      "sol"           Sol with no dense node (2026-09-04, the owner's "maybe
                      it's better to try without sage at all"): Sol as
                      shipped over ComfyUI's stock attention, so the steps
                      outside Sol's window and Sol's own fallback run stock.
                      On an armed server the probe's counterfactual becomes
                      stock attention.

    An image (single-frame) arm, a parked lane, carries what the archived
    image graphs carried: sage alone.
    """
    is_image = bool(extra.get("single_frame", False))
    dense = extra.get("dense_attn", False)
    dense_mode = ("none" if dense is True else dense) or None
    if dense_mode is not None and dense_mode not in _DENSE_ATTN_MODES:
        raise SystemExit(f"dense_attn={dense!r} is not one of {_DENSE_ATTN_MODES}")
    if dense_mode is None:
        if is_image:
            return True, False, None
        dense_mode = DENSE_CHAINS[DEFAULT_DENSE_CHAIN]["dense_attn"]
    if dense_mode in ("sol", "ck", "sage_sol"):
        if is_image:
            raise SystemExit(f"dense_attn={dense_mode!r} names a Sol-over-dense video arm; "
                             "it cannot be an image arm")
        return dense_mode == "sage_sol", bool(extra.get("sol_on", True)), dense_mode
    return dense_mode == "sage", False, dense_mode


def _takes_default_chain(extra: dict) -> bool:
    """True for a GRAPHS entry whose dense kernel is the default's to choose:
    a video entry that names no `dense_attn`. Everything
    else is an arm about a particular chain and stays on it under `--chain`."""
    return (not extra.get("single_frame", False)
            and not extra.get("dense_attn", False))


def _on_chain(extra: dict, chain: str) -> dict:
    """A default-taking entry's extras, moved onto `chain`.

    The chain's keys from `h3_config.DENSE_CHAINS` are laid under the entry's
    own, so an entry that already names a `sage_mode` keeps it."""
    return {**DENSE_CHAINS[chain], **extra}


def _suffix_output_prefixes(wf: dict, chain: str) -> None:
    """Append `_<chain>` to every `filename_prefix` in a built graph, in place,
    so a clip from the alternate set cannot land on top of the shipped
    graph's. Done on the built graph because the builder derives a default
    prefix per task when the entry names none."""
    for node in wf.values():
        prefix = node.get("inputs", {}).get("filename_prefix")
        if isinstance(prefix, str):
            node["inputs"]["filename_prefix"] = f"{prefix}_{chain}"


def _is_distill_experiment(fname: str, extra: dict) -> bool:
    """A graph from the distill experiments (2026-09-25 on), not a shipped pick.

    Derived like the image split was: every `_savelat` or `_x0` twin (the
    experiments' instrumentation), and every `h3_probe_*` graph that runs a
    distill, whether a PDD or FlashGen LoRA, FastH3's checkpoint, a step
    switch or an audio refine pass. Entries that are experiments without being
    probes say so with `distill_experiment=True`. The shipped distill graphs
    (`h3_text_to_video_pdd`, `_flashgen`, the PDD ref and fl2v graphs) stay at
    the root. The owner, 2026-09-27: "you and fastdude's modified/new
    workflows can go into a new subfolder called distill_experiments".
    """
    stem = fname.removesuffix(".json")
    if extra.get("distill_experiment") or stem.endswith(("_savelat", "_x0")):
        return True
    if not stem.startswith("h3_probe_"):
        return False
    lora = str((extra.get("lora") or ("",))[0]).lower()
    return bool(extra.get("pdd") or "pdd" in lora or "flashgen" in lora
                or "fasth3" in str(extra.get("unet", "")).lower()
                or extra.get("step_switch") or extra.get("audio_refine"))


def _graph_dir(out, extra: dict, fname: str = ""):
    """Which directory under `workflows/` a graph is written to.

    **Derived from `single_frame`, never declared per graph.** The split is by
    use case -- video at the root, the experimental image gen/edit path in
    `image/` -- and "renders one frame" is exactly what makes a graph an image
    graph. A separate `image=True` flag would be a second source of truth for
    one fact, and the two would eventually disagree; the failure would be a
    graph in the wrong folder, which is invisible until a check that walks one
    folder stops seeing it.

    `h3_config.GRAPH_DIRS` is the matching list on the reading side. If a third
    use case ever appears, both have to learn about it.
    """
    if extra.get("single_frame"):
        return out / SINGLE_FRAME_DIR
    if extra.get("daily"):
        # What the owner renders with (`h3_config.DAILY_GRAPHS`): the one
        # folder that is declared, since use cannot be derived from a graph.
        return out / DAILY_DIR
    return out / "distill_experiments" if _is_distill_experiment(fname, extra) else out


def _ref_image_slots(ref_images_on: bool, ref_image_count: int,
                     ref_images: tuple[str, ...] | None):
    """[(load_id, filename)] for the reference images a graph wires.

    `ref_images` names the files explicitly and sets the count from its own
    length, which is what the image graphs use -- a scene's references are part
    of what the scene IS, not a separate knob to keep in sync. Without it the
    count comes from `ref_image_count` and the files are the two placeholders,
    which is what every video graph has always done.
    """
    if not ref_images_on:
        return []
    placeholders = [PLACEHOLDER_IMAGE_A, PLACEHOLDER_IMAGE_B]
    if ref_images is None and ref_image_count > len(placeholders):
        # `[A, B][:3]` is 2 files, not an error, so without this a graph asking
        # for 3 placeholder references silently wires 2. That lands as a
        # check_ref_prompt_labels failure much later, naming the prompt rather
        # than the count that caused it. Ask for explicit `ref_images` instead:
        # a third placeholder would have to be chosen here, sight unseen, and
        # the role prose in _IMAGE_ROLE_PROSE is the caller's to declare.
        raise SystemExit(
            f"ref_image_count={ref_image_count} but only {len(placeholders)} "
            "placeholder images exist. Pass `ref_images=(...)` naming the "
            "files, so the graph declares what it wires.")
    files = (list(ref_images) if ref_images is not None
             else placeholders[:ref_image_count])
    if not 1 <= len(files) <= len(_REF_IMAGE_NODES):
        raise SystemExit(
            f"{len(files)} reference images: the generator reserves "
            f"{len(_REF_IMAGE_NODES)} stable loader slots and the same "
            "number of image positions in its typed append-id budget.")
    return list(zip(_REF_IMAGE_NODES, files))


def _append_image_inputs(load_id: str, chain, ref_upscale: bool, ref_qwen_short_edge: int) -> dict:
    """API inputs of one `MiniMaxH3AppendRefImage`: its loader, the chain so far, its sizing.

    One spelling for every graph that appends a still -- the reference graphs
    and the song graph with references -- so the two cannot drift.

    `size_policy` is a DynamicCombo since 2026-08-27, so its members are
    spelled DOTTED in the API form -- `size_policy.dit_short_edge`, never the
    flat `short_edge`, which the executor rejects. Same rule as
    `MiniMaxH3Resolution`'s `shape.wide_resolution`. They exist only under
    `max`; nothing emits them for `match`.

    `qwen_view` is a DynamicCombo since 2026-08-31, replacing an Int whose 0
    meant "no separate view". Dotted members again, and the size exists only
    under `separate` -- emitting it under `shared` is what the old flat form
    did and is exactly the unreachable-input state the combo removes. The
    SELECTION is always written: any shared-view arm must state its choice
    rather than inherit a node default that can move underneath it and
    silently retune a comparison.
    """
    inputs = {"image": [load_id, 0], "size_policy": "max"}
    if chain is not None:
        inputs["references"] = chain
    inputs["size_policy.dit_short_edge"] = _ref_short_edge()
    inputs["size_policy.allow_upscale"] = ref_upscale
    if ref_qwen_short_edge:
        inputs["qwen_view"] = "separate"
        inputs["qwen_view.qwen_short_edge"] = ref_qwen_short_edge
    else:
        inputs["qwen_view"] = "shared"
    return inputs


T2V_PROMPT = _bank_prompt("t2va_lighthouse")

I2V_PROMPT = _bank_prompt("i2va_lighthouse_keyframe")

#: The fl2va graphs' configured canvas. 3:2 at the 768 short edge the 4-step
#: turbo LoRA is named for, and `h3_config`'s `fast` row -- 864 tokens a frame
#: against 1008 for 16:9. It is a FALLBACK: under `from_keyframe` the canvas
#: comes from the loaded first frame and this governs only under `explicit`.
FL2V_CANVAS = dict(width=1152, height=768)

#: The final shot index the fl2va alignment sentence names. One, because the
#: body below is one continuous shot; raise it with the body, never alone.
FL2V_FINAL_SHOT = 1


def fl2v_prompt(length: int) -> str:
    """The fl2va prompt, with the alignment line resolved against `length`.

    **A function, not a constant, because the FL2VA alignment sentence carries
    two placeholders the other two modes do not.** `base_en.md:25` gives the
    string with `Shot N` and `S.SS` in it; N is the index of the actual final
    shot and S.SS is the effective duration to exactly two decimals. Typing a
    duration here would be a number that silently disagrees with the graph the
    moment `length` changes, so it is derived from the snapped frame count --
    the same grid `MiniMaxH3Conditioning` applies.

    **Note the punctuation.** FL2VA is the one alignment sentence of the three
    that carries no angle brackets and no square brackets: `Picture 1 (from
    Shot 1)`, not `<Picture 1> (from [Shot 1])`. I2VA and L2VA both bracket.
    `base_en.md:14-31` gives all three and this differs from its neighbours by
    exactly that, which is how a writer borrowing the I2VA form gets it wrong
    and nothing goes red -- preflight checks that the preamble names a Picture,
    not that it is the right sentence for the mode.

    One shot, deliberately. `base_en.md:62` says FL2VA "generally favors a
    single shot so the model can interpolate continuously from the first frame
    to the last", and that multiple shots are for when they are explicitly
    specified. So N is 1 here, and stays 1 unless the body grows a cut.

    **The body lives in `prompt_bank/fl2va_interior_converge.txt` since
    2026-09-03**, with the alignment line in it resolved at the frame count
    the manifest declares; only the sentence below is still built here, and
    `_retimed_from_bank` asserts the two agree at that count.
    """
    return _retimed_from_bank(FL2V_PROMPT_ID, _fl2v_alignment, length)


FL2V_PROMPT_ID = "fl2va_interior_converge"


def _fl2v_alignment(seconds: float) -> str:
    return (
        "How the reference pictures align with the target video \u2014 Picture 1 "
        "(from Shot 1) aligns with the 0.00-second mark of the target video; "
        f"Picture 2 (from Shot {FL2V_FINAL_SHOT}) aligns with the {seconds:.2f}-second "
        "mark of the target video.")



#: The final shot index the L2VA alignment sentence names. One, for the same
#: reason `FL2V_FINAL_SHOT` is one: a single continuous shot is what lets the
#: model converge on the closing frame.
L2V_FINAL_SHOT = 1


def l2v_prompt(length: int) -> str:
    """The l2va prompt, with the alignment line resolved against `length`.

    **L2VA is the mode with no coverage until now**, and its alignment sentence
    is the third of base_en's three. It is NOT the fl2va one with a field
    removed: `base_en.md:31` brackets BOTH the label and the shot --
    `<Picture 1> (from [Shot N])` -- where fl2va brackets neither. Getting that
    wrong is not cosmetic, because `preflight_graph._expected_base_alignment`
    compares the preamble to the guide's own template by exact string.

    `Shot N` and `S.SS` are placeholders and are resolved here, for the same
    reason `fl2v_prompt` resolves them: a duration typed in as a literal
    disagrees with the graph the moment `length` moves.

    One shot, and the body converges rather than interpolating between two
    endpoints -- there is only one endpoint. The opening is unconstrained,
    which is the whole difference from fl2va, so the body must not imply a
    starting frame the model was never given.

    **The body lives in `prompt_bank/l2va_interior_converge.txt` since
    2026-09-03**, on the same terms as its fl2va twin above.
    """
    return _retimed_from_bank(L2V_PROMPT_ID, _l2v_alignment, length)


L2V_PROMPT_ID = "l2va_interior_converge"


def _l2v_alignment(seconds: float) -> str:
    return (
        "How the reference pictures align with the target video \u2014 "
        f"<Picture 1> (from [Shot {L2V_FINAL_SHOT}]) aligns with the "
        f"{seconds:.2f}-second mark of the target video.")

#: The market scene as a ref2va task: the same three shots and the same two
#: speakers, with the stallholder's appearance carried from a reference image.
#:
#: **This is not `LONG_T2V_PROMPT` with sections bolted on.** ref_en's format
#: differs from base_en's in four ways (ref 5.2) and three of them apply here:
#: the main field is `detailed_description`, the style opening sits BEFORE
#: `[Shot 1]` rather than after it, and the reference labels are inserted at
#: first appearance and where their roles apply.
#:
#: `<Picture 1>` is cited INSIDE the `<Subject 1>` definition and gets no
#: standalone entry, per ref 2.2: the image defines a character, it is not a
#: keyframe or composition anchor. `docs/prompting.md` 9.3 calls this the
#: single most-violated ref2va rule in this repo's shipped prompts.
#:
#: Camera motion is base 4.3 vocabulary, the same repair the t2v version got in
#: `d5be353` -- `trucks left`, `holds a static shot`, and no amplitude or speed
#: phrase at all, because the guide writes medium and normal by omitting them.
#: The porter's identity is given in [Shot 1] where he first appears (4.4).
#: The reference the market ref2va arms carry. One image, one subject.
def graph_length(extra: dict) -> int:
    """The frame count the build loops give a graph.

    Both loops spell this `{"length": LONG_LENGTH, **rest}` -- the shipped
    default is LONG, not `build_api`'s signature default, and an entry naming
    its own `length` wins. Factored out because `--dump-prompts` has to
    resolve it identically: `l2v_prompt`/`fl2v_prompt` bake the alignment
    TIMESTAMP from it, so a length resolved one way here and another way there
    yields two prompts differing only in a number -- exactly the silent
    disagreement the check that consumes it exists to detect.

    The bench stamped copies pass `length=` directly and are not built from a
    GRAPHS entry; they are outside `graph_paths` and so outside that check.
    """
    return extra.get("length", LONG_LENGTH)


def resolve_default_prompt(task: str, prompt: str | None, *,
                           length: int, last_frame: bool,
                           first_frame: bool) -> str:
    """The prompt a graph gets when its GRAPHS entry declares none.

    **THE DEFAULT PROMPT FOLLOWS THE SOCKETS, NOT THE TASK STRING.** `i2v`
    covers both keyframe modes -- one wired frame or two -- and they take
    DIFFERENT alignment sentences (`base_en.md:14-31`), so keying this on
    `task` alone hands an fl2va graph the I2VA line. Nothing downstream would
    catch it: preflight checks that the preamble names a Picture, not that it
    is the right sentence for the mode.

    **One copy, called from three places.** It stood as an identical
    five-line expression in `build_api` and `build_ui` until 2026-08-28, when
    `--dump-prompts` needed the same answer and would have made a third.
    `bench/check_ref_prompt_labels.py` compares shipped graphs against what
    this returns, so a fourth copy would be a check grading a graph against a
    restatement of the rule rather than against the rule.
    """
    if prompt is not None:
        return prompt
    if task == "i2v" and last_frame and not first_frame:
        return l2v_prompt(length)
    if task == "i2v" and last_frame:
        return fl2v_prompt(length)
    return {"t2v": T2V_PROMPT, "i2v": I2V_PROMPT, "r2v": R2V_PROMPT}[task]



#: **The DEMAND pair. Not a length pair, and the difference decides what it
#: tests.** Owner-requested 2026-08-28 as a length experiment and reframed the
#: same evening by his own correction: "what i meant by ghosted fruit was....
#: theres too much detail in whats being asked of the scene / frames being
#: generated." The axis is detail DEMANDED BY THE PROMPT -- a cause, readable
#: from the text -- not detail measured in the render, which is a result and is
#: ambiguous between a plain scene and a destroyed one.
#:
#: The long arm elaborates materials, light, surface and position. Under the
#: demand reading that is not a null manipulation dressed up as more words: it
#: is precisely asking for more fine structure to be resolved, which is the
#: thing under test. **So do not describe these as "same scene, more words".
#: They are "same scene, more detail demanded".**
#:
#: **PRE-REGISTERED, before either arm rendered.** Demand predicts the LONG arm
#: is worse than its short counterpart in BOTH scenes, with the damage
#: concentrated on the elaborated surfaces. A length-of-conditioning account
#: predicts no consistent direction. Two scenes agreeing is the result; one is
#: a fact about that scene.
#:
#: Note what this does NOT exclude. Some elaborations name small objects the
#: short arm does not -- a price card at each bin, a length of chain in a
#: waiting customer's hand. Under the demand hypothesis that IS the
#: manipulation rather than a confound, since a small object is demand. What is
#: excluded, and was the discipline throughout, is any new SUBJECT, ACTION,
#: CAMERA MOVE or LINE, none of which differ.
#:
#: Originally written after `h3_ref2v_scene_kitchen`
#: rendered roughly two shots where its prompt asked for four while
#: `h3_ref2v_scene_subway` honoured all three cuts -- the two differed in
#: description length among other things, and nothing isolated it.
#:
#: **Everything is held except word count.** Same scene, same three shots, same
#: cuts (the arms carried header times, 00:04.500 and 00:09.000, when this was
#: written; the bank text carries none, `docs/prompting.md` section 3.1), same two speakers, same four lines of dialogue
#: verbatim, same three camera moves verbatim, same subjects, same actions.
#: The long version ELABORATES existing content -- materials, surfaces, light,
#: position -- and introduces no new subject, action, camera move or line.
#: That discipline is the whole experiment: a longer prompt that also adds
#: content confounds length with complexity and answers nothing.
#:
#: **A hardware aisle, chosen for detail density rather than for looking good.**
#: `docs/eval_comparison.md` records that a scene which cannot express the
#: defect cannot discriminate it however it is scored, and that the candidate
#: axis is fine detail near the latent resolution limit rather than delta.
#: Bins of small fasteners and a pegboard wall are that, in a scene structurally
#: unlike the produce market so the two are not near-duplicates.
#:
#: **Deliberately NOT a marker arm.** No caption, no lyrics, no cutoff -- those
#: are `h3_ref2v_scene_*`'s job, and a marker here would be a second axis.
T2V_AISLE_SHORT = _bank_prompt("t2va_hardware_aisle_short")

#: The long arm. **Same content, elaborated.** Read the two side by side before
#: changing either: any edit that adds a subject, an action, a camera move or a
#: line to this one and not to the short one destroys the comparison.
T2V_AISLE_LONG = _bank_prompt("t2va_hardware_aisle_long")


#: **The second description-length pair, on a different scene.** Owner-requested
#: alongside the aisle pair: "a different scene entirely. not a market. just one
#: with the same type of shit happening in it."
#:
#: **Two scenes rather than one is the point.** A length effect seen on a single
#: scene is a fact about that scene. Two independent scenes carrying the same
#: manipulation is the difference between an anecdote and a result, and it costs
#: two more renders rather than a new design.
#:
#: A sorting line: mixed recyclables on a moving belt, four workers picking.
#: Chosen to match the aisle pair on what is being tested -- many small
#: high-contrast objects near the latent resolution limit, people moving through
#: a wide frame -- while sharing nothing of its setting, palette or lighting.
#: The belt also supplies continuous independent object motion, which the aisle
#: does not, so the pairs differ on that axis deliberately.
#:
#: Same discipline as the aisle: identical dialogue, camera moves, cuts,
#: shot structure and subjects across the two lengths. The long arm elaborates
#: and does not extend.
T2V_SORTLINE_SHORT = _bank_prompt("t2va_sortline_short")

#: The long arm of the sorting-line pair. **Same content, elaborated.**
T2V_SORTLINE_LONG = _bank_prompt("t2va_sortline_long")


#: **The PREDICTABILITY pair, and the sharpest discriminator built here.**
#: Owner-stated 2026-08-28, after both two-shot ablation arms came back clean:
#: "i bet even long prompts are fine. so long as you dont introduce like 5
#: unique shots and tons of shit changes everywhere like lights flashing
#: different led colors every second of every shot from different places and
#: people moving around or a handheld camera fight scene where the model cant
#: easily predict what comes next like it could say... a tracking shot of a
#: boxy house going from left to right across. steady and not changing position
#: at all, just moving left to right like its on a rail"
#:
#: **The axis is how hard the next frame is to predict, not how much it
#: changes.** That distinction is what every earlier account here missed. A
#: rail move translating the whole frame has near-maximal inter-frame delta and
#: near-zero uncertainty: everything is where it was, shifted. A handheld shot
#: of a crowd under changing coloured light has similar delta and no
#: extrapolable structure at all.
#:
#: **Both arms are LONG on purpose**, ~500 words, because the owner's claim has
#: two halves and this tests both: long prompts are fine, AND unpredictability
#: is what is not. If the rail arm comes back clean at this length, length is
#: exonerated in the same render that indicts churn.
#:
#: **Held: one shot each, no cuts, same canvas, same length, same settings, no
#: dialogue in either.** Shot count is deliberately equalised at ONE so this
#: does not re-run the ablation's confound -- pdd's arms move delta, demand and
#: shot content together, and that is why they could not separate anything.
#:
#: **What it predicts, registered before either rendered.** Delta says the rail
#: arm is worst, since a full-frame translation is the highest delta in this
#: repo. Predictability says the rail arm is CLEAN and churn breaks. They
#: cannot both be right, and the rail arm is cheap to judge because ghosting on
#: a rigid boxy building is unmissable.
T2V_RAIL_LONG = _bank_prompt("t2va_rail_dolly_long")

#: The churn arm. **Same length, same one-shot structure, same canvas. Every
#: source of unpredictability the owner named, in one shot.**
T2V_CHURN_LONG = _bank_prompt("t2va_crowd_churn_long")

MARKET_REF_IMAGES = ("dirk_runway2.jpeg",)

MARKET_REF2V_PROMPT = _bank_prompt("ref2va_market_stallholder")


R2V_PROMPT = _bank_prompt("ref2va_image_ref_default")


# --------------------------------------------------------------------------
# API format
# --------------------------------------------------------------------------

sys.path.insert(0, str(HERE.parent))
from h3_rules import (  # noqa: E402
    aspect_in_range, describe_aspect_range, describe_length,
    duration_in_range, duration_of, is_single_frame, max_legal_length,
    min_legal_length, snap_length,
)


def _core_cpu_when_no_card():
    """With no CUDA device visible, put ComfyUI core on its CPU path before anything imports it.

    Importing `comfy_extras.nodes_minimax_h3` pulls `comfy.model_management`,
    which picks a torch device at import and raises "No CUDA GPUs are
    available" when `CUDA_VISIBLE_DEVICES=` hides the card. Masking is how this
    script runs while another process renders (`docs/checks.md`, "While a
    render is on the card"; on 2026-09-04 the context creation failed against
    a busy card). The switch lived inside `_ref_short_edge` alone until
    2026-09-14, but `_resolution_widgets` reaches core first, through
    `resolution.py`, so a masked build raised there (seen at `1345791` and at
    the head of that day). Call this before every path into core, with
    ComfyUI's root already on `sys.path`; with a card visible it changes
    nothing.
    """
    import torch
    if not torch.cuda.is_available():
        import comfy.cli_args
        comfy.cli_args.args.cpu = True


def _song_frames_read(seconds: float, window_frames: int, context_frames: int, timeline: str = "") -> int:
    """How many frames of its track the song node reads for an extent of `seconds`.

    The node's own count, from the node's own planner (`loop_plan.frames_read`), with the
    total it derives from the extent (`audio_freeze_song.py::execute`, `total_frames`): a
    source loader's cap is set from it, so the graph loads what the plan reads and the mask's
    tracker works on nothing else. `loop_plan.py` is loaded by path as `resolution.py` is
    below, and for the same reason; its one import outside the standard library is core's
    frame rate.
    """
    import importlib.util

    for extra in (HERE.parent.parent.parent, HERE.parent):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    _core_cpu_when_no_card()
    spec = importlib.util.spec_from_file_location("_h3_loop_plan_for_build", HERE.parent / "loop_plan.py")
    plan = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plan       # its dataclasses look their module up by name
    spec.loader.exec_module(plan)
    return plan.frames_read(int(math.ceil(float(seconds) * FPS)), int(window_frames), int(context_frames), timeline)


def _resolution_widgets(width, height, length):
    """The Resolution node's inputs for an explicit width/height.

    Reverse of what the node does: find which band holds this resolution and
    which option label names it, so a graph asking for 1344x768 selects the
    entry that says what it costs rather than typing two numbers that say
    nothing. Falls back to `custom` for anything outside the trained family,
    which the node then reports as outside rather than refusing.
    """
    # Load resolution.py by path rather than as a package member: importing
    # the package runs its __init__ and nodes.py, which need comfy_api. The
    # module's own imports need ComfyUI's root (comfy_api) and this repo's
    # root (h3_rules), both of which this script otherwise runs without.
    import importlib.util

    for extra in (HERE.parent.parent.parent, HERE.parent):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    # resolution.py imports core to sweep the canvases: the first path into it
    _core_cpu_when_no_card()
    spec = importlib.util.spec_from_file_location(
        "_h3_resolution_for_build", HERE.parent / "resolution.py")
    res = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(res)

    # DynamicCombo members are addressed by their DOTTED path in the API form:
    # `shape.wide_resolution`, not `wide_resolution`. The flat spelling was
    # what this emitted until 2026-08-13, and ComfyUI's executor rejects it
    # with `required_input_missing` naming `shape.wide_resolution` -- so every
    # API graph in this repo was unsubmittable, which is the form the benches
    # drive. Our own `validate_api` accepted it, which is why nobody noticed:
    # it was checking a shape ComfyUI does not use. Found by running
    # `bench/smoke_h3.py` against a live server, not by any check.
    #
    # Both spellings were tried against a running ComfyUI before this changed;
    # dotted is accepted and flat is refused, for the band case and the custom
    # case alike.
    for band, entries in res._resolutions().items():
        if (width, height) in entries:
            return {"shape": band,
                    f"shape.{band}_resolution": res._label(width, height),
                    "length": length}
    return {"shape": "custom", "shape.width": width, "shape.height": height,
            "length": length}


def _require_core_encoder(name: str) -> str:
    """Refuse an encoder file no loader in this pack can open.

    `h3_config.CORE_LOADED_ENCODERS` is the set core's `CLIPLoader` (and so
    `MiniMaxH3EncoderLoader`) loads. Anything else appears in core's menu by
    filesystem discovery and fails at load; the adapter that once opened
    compressed-tensors artifacts was deleted with its lane on 2026-09-13.
    """
    if name not in CORE_LOADED_ENCODERS:
        raise SystemExit(
            f"{name!r} is not a ComfyUI-native H3 encoder "
            f"(h3_config.CORE_LOADED_ENCODERS); no loader in this pack opens it")
    return name


def _ref_short_edge():
    """ComfyUI's reference short edge, read rather than repeated.

    `MiniMaxH3AppendRefImage` defaults this input to core's
    `REF_IMAGE_SHORT_EDGE`. Writing 2048 into the graph as a literal would be
    a second place to edit that agrees with the first only by inspection --
    if core ever moves the constant, the node's default moves and the shipped
    graphs quietly do not. There is no test that could tell those apart,
    because a duplicated decision has no observable disagreement until the
    day it disagrees.
    """
    # This script is designed to run without ComfyUI importable -- it
    # validates over HTTP -- so put the root on the path just for this, and
    # at position 0: the repo root is already there (`sys.path.insert` near
    # the top), and `nodes_minimax_h3` does a bare `import nodes` that finds
    # THIS PACK's `nodes.py` if the repo wins. That is the `import nodes`
    # trap in `docs/comfy_notes.md`.
    #
    # **Found by marker, not by counting `..`.** It was `HERE.parent.parent
    # .parent` until 2026-09-03, which is the ComfyUI root only when the pack
    # sits directly in `custom_nodes/`; from a git worktree of this repo it
    # resolves three levels short and the generator cannot run at all. The
    # walk returns the same directory in the ordinary layout.
    # **Not finding one is not an error here.** A caller can have made the
    # module importable already -- `bench/check_generator_constants.py`
    # imports core BEFORE this module for exactly the shadowing reason above,
    # so by the time this runs it is a `sys.modules` hit and no path work is
    # needed. Raising when the walk comes up empty made that green check go
    # red, which is the "a correct fix moves where a constraint applies"
    # rule in CLAUDE.md: let the import decide, and it raises on its own if
    # ComfyUI really is unreachable.
    root = next((p for p in HERE.parents
                 if (p / "comfy_extras" / "nodes_minimax_h3.py").is_file()), None)
    if root is not None and sys.path[0] != str(root):
        sys.path.insert(0, str(root))
    # The import pulls `comfy.model_management`; this function needs one
    # constant and no device (`_core_cpu_when_no_card`).
    _core_cpu_when_no_card()
    from comfy_extras.nodes_minimax_h3 import REF_IMAGE_SHORT_EDGE

    return REF_IMAGE_SHORT_EDGE


def _check_single_frame(single_frame, length):
    """`single_frame` is a property of the LENGTH; they may not disagree.

    Shared by both builders. Passing one without the other produces a graph
    that loads the one-frame VAE and renders five frames, or renders one frame
    and decodes it with the video decoder -- both silent, both wrong, and
    neither visible until someone looks at the pixels.
    """
    if single_frame != is_single_frame(length):
        raise SystemExit(
            f"single_frame={single_frame} with length={length}: the "
            f"single-image path is length=1 and nothing else. Set both or "
            f"neither.")


def _check_geometry(length, canvas):
    """Refuse to emit a graph the reference would reject.

    **Scope note, since 2026-08-13.** `canvas_mode` now defaults to
    `match_keyframe`, under which `MiniMaxH3KeyframeCanvas` derives the canvas
    from the loaded keyframe and the width/height in the graph are inert. So
    for an i2v graph the aspect assertion below validates the *configured*
    fallback, not what will render: swap in a 3:4 still and the graph renders
    768x1344, the most expensive canvas on the area cap, having passed a check
    that looked at 1344x768.

    That is not a hole, but it is a relocation worth naming. The aspect
    guarantee moves from build time to run time, where the node enforces it on
    the *source image* and raises -- which is where the reference enforces it
    too (`resolve_canvas_size`, called on `keyframes[0].size`). The check here
    still earns its place because the fallback matters the moment someone
    switches the mode back.

    This config shipped 362 frames for a week. It is on the 17n+5 grid, it is
    inside ComfyUI's own 3600 limit, and it renders -- it is just 15.083s
    against a 15s ceiling the reference enforces and ComfyUI does not. Nothing
    in the pipeline said so, which is exactly the failure this repo exists to
    make loud, so the generator now holds the rule rather than a comment.
    """
    cv = dict(CANVAS, **canvas)
    # length=1 is the single-image edit mode, not a very short video, so the
    # duration window does not apply and refusing it here would block the one
    # graph that wants it. The aspect rule below still applies -- that one is
    # about the canvas, which a single frame has exactly like a clip does.
    if not is_single_frame(length) and not duration_in_range(length):
        raise SystemExit(
            f"length {describe_length(length)} is outside MiniMax H3's 5-15s "
            f"window; legal counts are {min_legal_length()}-{max_legal_length()} "
            f"on the 17n+5 grid. Fix LONG_LENGTH/LENGTH in h3_config.py."
        )
    if not aspect_in_range(cv["width"], cv["height"]):
        raise SystemExit(
            f"canvas {cv['width']}x{cv['height']} is aspect "
            f"{cv['width'] / cv['height']:.3g}, outside H3's trained "
            f"{describe_aspect_range()} range."
        )


def _plain_model_chain(g, *, sage, sol, shift, head_chunks, dense_backend=None):
    """A second model path off the same UNETLoader, WITHOUT the LoRA.

    The two-stage split runs a different model on each half, so it needs two
    chains. This mirrors the primary chain built inline in `build_api` -- see
    the comments there for why each node sits where it does -- with ids in the
    40s and one difference: no `LoraLoaderModelOnly`.

    **The shift must be identical on both.** Both halves read sigmas from one
    `BasicScheduler`, and the shift is what that schedule is built from; two
    different shifts would mean the two halves are integrating different
    curves and the handoff is meaningless.
    """
    src = ["1", 0]
    g["40"] = {"class_type": "MiniMaxH3SigmaShift",
               "inputs": {"model": src,
                          **(shift if shift is not None else SIGMA_SHIFT)}}
    src = ["40", 0]
    if dense_backend is not None:
        # The primary chain's backend node (id 58) mirrored, so both halves of
        # a split run the same dense kernel. Node id 59.
        g["59"] = {"class_type": "ModelAttentionBackend",
                   "inputs": {"model": src, "attention": dense_backend}}
        src = ["59", 0]
    if sage:
        g["41"] = {"class_type": "MiniMaxH3SageAttention",
                   "inputs": {"model": src, **dict(
                       SAGE_NODE,
                       **({} if head_chunks is None
                          else {"head_chunks": head_chunks}))}}
        src = ["41", 0]
    if sol is not None:
        g["42"] = {"class_type": SOL_NODE,
                   "inputs": {"model": src, **sol_api_inputs(sol)}}
        src = ["42", 0]
    return src


def build_api(task: str, *, sage: bool = True, prompt: str | None = None,
              length: int = LENGTH, seed: int = SEED,
              sol: dict | None = None, sol_impl: str = "ours",
              canvas_mode: str = "match_keyframe",
              last_frame: bool = False,
              first_frame: bool = True,
              stamp: bool = False, unet: str | None = None,
              lora: tuple[str, float] | None = None,
              # Apply `lora` at the call (lora_branch.py) instead of merging
              # it into the int8 weight, where a sub-step delta is lost. On by
              # default since 0.154.0 (owner, 2026-09-26: every small LoRA on
              # int8 goes through our branch); False only on a merge control.
              lora_branch: bool = True,
              # MiniMaxH3LoRABranch's `blocks`: "all", or a range like "34-49".
              # FlashGen on its late blocks alone keeps the 4-step finish with
              # about half the haze (fastdude's FT1, 2026-09-27).
              lora_blocks: str = "all",
              steps: int | None = None, shift: dict | None = None,
              sampler_name: str | None = None, scheduler_name: str | None = None,
              head_chunks: int | None = None,
              # Block-49 probes (docs/h3_block49_quant_error.md, 2026-09-15).
              # sage_mode overrides SAGE_NODE's mode ("fp8++ balanced" turns
              # on the fork's qk_balance); channel_balance is
              # MiniMaxH3ChannelBalance's combo value, placed before the
              # attention nodes since it only patches norm weights;
              # exact_blocks is MiniMaxH3ExactBlocks' list, placed after the
              # attention nodes (its forward survives either order).
              sage_mode: str | None = None,
              channel_balance: str | None = None,
              exact_blocks: str | None = None,
              # dense_backend: ComfyUI core's ModelAttentionBackend value, put
              # where the Sage node would sit (dense_attn="ck").
              dense_backend: str | None = None,
              # Owner decision 2026-09-13: True, the node's own default and
              # what sglang, diffusers and DiffSynth do (every still to the
              # 2048 short edge, one copy for both towers). It was flipped
              # False on 2026-08-28 for cost; the cost is now shown by
              # the conditioner's preview before a render rather than
              # avoided by default. `REF_VIDEO_BUDGET` still turns it off on
              # the video-bearing arms, for memory.
              ref_upscale: bool = True,
              manual_sigmas: str | None = None,
              ref_video_policy: str = "comfy",
              ref_image_policy: str = "comfy",
              # A size selects `qwen_view = separate` at that short edge: the
              # node default and the owner's choice since 2026-10-03. 0 is
              # `shared`, one copy for both towers, the serving
              # implementations' behaviour and the default from 2026-09-13;
              # the instruments pass it (h3_config.REF_QWEN_SHORT_EDGE says
              # which).
              ref_qwen_short_edge: int = REF_QWEN_SHORT_EDGE,
              ref_video: bool = False, ref_video_audio: bool = True,
              ref_images_on: bool = True, ref_image_count: int = 2,
              ref_images: tuple[str, ...] | None = None,
              pdd: bool = False,
              pdd_heads: bool = True,
              # 0 everywhere, and it should stay that way: the node reads
              # the step count off the sampler's own sigma schedule at run
              # time. The three 4-step arms carried 4 here until
              # 2026-08-27, which made the step count a fact typed in two
              # places with only a warning between them. Non-zero now
              # FORCES uniform blocks and ignores the schedule, which is an
              # off-schedule experiment, not a step-count setting.
              pdd_nfe: int = 0,
              # MiniMaxH3PDDLoRA's `backbone_apply`: the exact branch by
              # default, "merge" only on the control arm (0.154.0).
              pdd_backbone_apply: str = "exact branch",
              ref_audio: bool = False,
              split_at: int | None = None,
              split_base_last: bool = True,
              single_frame: bool = False,
              cache: dict | None = None,
              vae_encoder: str | None = None,
              clip: str | None = None,
              # Reference pathway knobs, 2026-09-03. `ref_latents=False`
              # leaves both VAE sockets of the reference conditioner
              # unwired, which since ComfyUI PR 16065 (core) and 0.99.33
              # (ours) means every reference conditions the text encoder
              # only and adds no rows to the DiT. `native_ref=True` emits
              # core's MiniMaxH3ReferenceToVideo in place of the typed chain,
              # for the A/B against our conditioner; stills only, API only.
              ref_latents: bool = True,
              native_ref: bool = False,
              # The audio-freeze lane (docs/h3_audio_freeze.md): a known track
              # written into the target audio rows and frozen with a nested
              # noise_mask, between the preflight and the sampler. The muxer
              # takes the node's own slice of the track; the audio decoder is
              # removed because the frozen latent is a control signal and its
              # round trip is lossy. `freeze_mask` 0 is the hard freeze.
              freeze_audio: bool = False, freeze_start: float = 0.0,
              freeze_mask: float = 0.0, freeze_track: str = PLACEHOLDER_AUDIO,
              # The loop's first seam, API only: two windows of the track,
              # the second taking the first's sampled latent as frozen
              # context (docs/h3_audio_freeze.md section 4 step 6).
              freeze_windows: int = 0, freeze_context: int = 39,
              # Guide audio on top of the freeze (docs/h3_audio_freeze.md
              # idea 6 as the fl2va-base hybrid): core's MiniMaxH3AddGuide
              # anchors the same track as conditioning rows at frame 0, so
              # the track sits in the sequence twice, frozen in register and
              # as a reference-style block. Needs freeze_audio.
              freeze_guide: bool = False,
              # The audio attention gain node in front of the guider, inert at
              # (1.0, 1.0) so a bench arm can patch its values. API only.
              freeze_gain: bool = False,
              # The whole-track node in place of the conditioner, sampler,
              # decoders and muxer: windows planned from the track. The
              # shipped graph caps the plan at `freeze_song_seconds` so a
              # first run is a quick look; None covers the whole track.
              freeze_song: bool = False, freeze_song_seconds: float | None = 30.0,
              # `mm:ss label` lines the song node lines its windows up with;
              # "" is no timeline.
              freeze_song_timeline: str = "",
              # Reference stills for the song node, one Append Ref Image each,
              # in <Picture N> order.
              freeze_song_refs: tuple[str, ...] | None = None,
              freeze_song_lists: tuple[tuple[str, str, str, int], ...] | None = None,
              # Masked video-to-video on the song node (`video_mask.py`): the
              # track comes from a source video, the Subject Track node finds
              # one subject across the clip's cuts from a phrase and one pick
              # (`subject_track.py`), and each window keeps the source outside
              # that mask. The loader's frame cap is the song's extent plus
              # one window, which is as far past the track as the plan can
              # reach.
              freeze_song_source: bool = False,
              # The Masked Source's widget values when `freeze_song_source` is
              # on; None is `h3_config.MASKED_SOURCE` (the shipped render with
              # no motion reference), `MASKED_MOTION_SOURCE` the ref2va motion
              # graph's. One dict, so a graph cannot half-override the node.
              masked_source: dict | None = None,
              # The Masked Prompt node's widget values (`masked_prompt.py`,
              # `h3_config.MASKED_PROMPT`): the song node's prompt is then that
              # node's output, and `prompt` must be the text it writes for this
              # graph, so the bank holds a copy of what every shipped graph
              # renders. None leaves the prompt typed into the song node.
              masked_prompt: dict | None = None,
              # Sapiens2's part mask into the Masked Source's `parts`
              # (`sapiens2_parts.py`, `h3_config.SAPIENS2` and
              # `SUBJECT_PARTS`); `masked_source` then has to choose
              # `the wired parts`.
              masked_parts: bool = False,
              # The look before a render: the song node on `preview`, which
              # samples nothing and loads no model, with the tracker's tiles
              # and shot table saved and the Masked Source's preview shown.
              # The mask it tracks is kept, so the render graphs reuse it.
              masked_review: bool = False,
              # Audio-only refinement after the pass (audio_refine.py,
              # h3_config.AUDIO_REFINE): the sampled latent's video frozen and
              # its audio reopened, then a partial-denoise pass on the model
              # from BEFORE the LoRA, on its own copy of the base chain.
              audio_refine: bool = False,
              # The refine pass's frozen-video cache (frozen_video_cache.py,
              # h3_config.FROZEN_VIDEO_CACHE) on the refine model, node 88.
              refine_cache: bool = False,
              # Core's BlockSparseAttention at these API inputs, in Sol's slot
              # (node 21). For a checkpoint trained with VSA whose upstream
              # recipe is core's node (FastH3, h3_config.FASTH3_CORE_VSA).
              core_vsa: dict | None = None,
              # Scouting: the video decoded by h3_config.DRAFT_VAE, and the
              # latent the decoders read saved in two halves (video, audio) so
              # a keeper is decoded for real by h3_decode_saved_latent_api.json.
              # Audio decodes on the real audio VAE either way. Nodes 100-103
              # (90-97 are the prompt lists).
              draft_decode: bool = False,
              # Save the latent the decoders read, in two halves (video,
              # audio), without changing the decode: the `_savelat` probe twins
              # of the 2026-09-26 distill run. Never on a shipped graph (every
              # render would write ~40 MB). Nodes 101-103.
              save_latents: bool = False,
              # The frozen-row test (docs/research/2026-09-26_distill_routing.md,
              # "The frozen-row test"): on an audio_refine graph, observe the
              # denoise mask the refine pass uses (MiniMaxH3DenoiseMaskProbe)
              # and save pass 1's latent and the refine pass's denoised output
              # beside the final one, all in one execution. Nodes 106-112.
              probe_frozen_rows: bool = False,
              # Route 3 of the distill-routing idea: FlashGen's first two steps
              # through MiniMaxH3LoRABranch, then PDD8's finish from FlashGen's
              # own endpoint, DisableNoise on pass 2, one shift on both
              # (h3_config.STEP_SWITCH_*). Nodes 120-123 and the PDD chain at
              # +200. Pass 1 is this call's own graph; set its FlashGen knobs.
              step_switch: bool = False,
              # What pass 2 is: "pdd8" (route 3, FlashGen then PDD8's finish
              # at STEP_SWITCH_PASS2_SIGMAS) or "flashgen" (the reverse: this
              # graph's own PDD chain, then FlashGen at step_switch_sigmas), or
              # "base" (this graph's PDD chain, then the undistilled base on
              # Euler at step_switch_sigmas; open_experiments #37), or
              # "fasth3" (this graph's PDD chain, then FastH3's own checkpoint
              # at step_switch_sigmas and step_switch_shift, on its core VSA
              # at the contract; h3_config.STEP_SWITCH_FASTH3).
              step_switch_to: str = "pdd8",
              step_switch_sigmas: str | None = None,
              step_switch_shift: dict | None = None,
              # The FlashGen pass's `blocks` in a PDD-first switch.
              step_switch_blocks: str = "all",
              # Save every step's x0 prediction (MiniMaxH3StepX0Observer, node
              # 113) on the main pass's model, for finding the step where a
              # moving person first appears twice (docs/h3_distills.md). About
              # 40 MB a step: probe twins only.
              probe_step_x0: bool = False,
              # Capture on core's sparse producer path (MiniMaxH3CoreSparseCapture,
              # node 114, after the attention node): inert unless H3_CAPTURE is set
              # with pre=; for open_experiments #45 on FastH3's contract graph.
              probe_core_sparse_capture: bool = False,
              out_prefix: str | None = None, **canvas) -> dict:
    """API-format graph, submittable as {"prompt": <this>} to POST /prompt.

    Node ids match `bench/bench_e2e_h3.py` so a timing run and a hand-edited
    graph can be compared node-for-node; "10" is the sampler in every graph.

    `unet` overrides the checkpoint the task would otherwise pick, for the
    probes that need a model source no task name describes. `lora` is
    (name, strength) and inserts `MiniMaxH3LoRABranch`, or LoraLoaderModelOnly
    on a merge control (`lora_branch=False`).
    """
    if task not in ("t2v", "i2v", "r2v"):
        raise ValueError(task)
    if ref_video_policy not in ("comfy", "release", "encoder"):
        raise ValueError(
            f"unknown ref_video_policy {ref_video_policy!r}; "
            "expected 'comfy', 'release', or 'encoder'"
        )
    if ref_image_policy not in ("comfy", "release", "encoder"):
        raise ValueError(
            f"unknown ref_image_policy {ref_image_policy!r}; "
            "expected 'comfy', 'release', or 'encoder'"
        )
    _check_single_frame(single_frame, length)
    if single_frame and (stamp or split_at):
        # Both reach for node 12, which the single-frame path deletes. Not
        # reachable from GRAPHS, but `build_api` is a public entry the benches
        # drive, and the failure would otherwise be a bare KeyError from a
        # dict literal rather than a sentence naming the combination.
        raise SystemExit(
            "single_frame does not compose with stamp or split_at: both wire "
            "the audio decoder (node 12), which the one-frame path removes "
            "because a single frame's audio is 0.04s of nothing.")
    _check_geometry(length, canvas)
    ref = task == "r2v"
    cv = dict(CANVAS, **canvas)
    prompt = resolve_default_prompt(task, prompt, length=length,
                                    last_frame=last_frame,
                                    first_frame=first_frame)

    _encoder = clip or MODELS["clip"]
    # Resolved once. `_resolved_steps` reaches BasicScheduler AND, on a PDD
    # graph, MiniMaxH3PDDLoRA's own `steps` -- the two must never be able to
    # disagree, which is exactly the class of bug this rewiring exists to make
    # unexpressible.
    _resolved_steps = steps if steps is not None else SAMPLING["steps"]
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet or MODELS["unet_ref2va" if ref else "unet_fl2va"],
                         "weight_dtype": "default"}},
        # Every encoder goes through `MiniMaxH3EncoderLoader`: core's own
        # load plus the two checks core does not do -- the checkpoint must
        # exactly populate the model, and the tokenizer must realise the
        # released special-token ids. Preprocessing is bit-for-bit what plain
        # `CLIPLoader` gives. The AWQ adapter branch that used to sit here
        # went with its lane on 2026-09-13; `_require_core_encoder` is what
        # refuses a file core cannot open, since no loader here opens it now.
        # `device` is core's `CLIPLoader` input, appended to our loader on
        # 2026-09-19 (CHANGELOG 0.131.0): "default" is ComfyUI's own
        # text-encoder device, which is the GPU whenever dynamic VRAM is on
        # (`comfy/model_management.py::text_encoder_device`). Written out so
        # a render record carries the placement rather than implying it.
        "2": {"class_type": "MiniMaxH3EncoderLoader",
              "inputs": {"encoder_name": _require_core_encoder(_encoder),
                         "device": "default"}},
        # The image VAE ONLY on the single-frame path. See h3_config: same
        # frozen encoder, decoder retrained for one temporal latent, and its
        # own README says it regresses multi-frame reconstruction -- so this
        # swap must never be reachable from a graph that renders a clip.
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": IMAGE_VAE if single_frame
                         else MODELS["video_vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["audio_vae"]}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "7": {"class_type": "KSamplerSelect",
              "inputs": {"sampler_name": sampler_name or _distill(lora, pdd, "sampler")}},
        "8": {"class_type": "BasicScheduler",
              "inputs": {"model": None,
                         "scheduler": scheduler_name or _distill(lora, pdd, "scheduler"),
                         "steps": _resolved_steps,
                         "denoise": SAMPLING["denoise"]}},
        "9": {"class_type": "BasicGuider",
              "inputs": {"model": None, "conditioning": ["5", 0]}},
        "10": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0],
                          "sigmas": None, "latent_image": ["5", 1]}},
        # Both decoders read the same packed AV latent and each pulls out its
        # own half; this is not a mistake in the wiring.
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        # VHS_VideoCombine instead of CreateVideo -> SaveVideo: one node, and
        # it muxes the audio itself. Node id 13; 14 is retired with SaveVideo.
        # The format sub-widgets (pix_fmt/crf/save_metadata/trim_to_audio) are
        # h264-mp4's own, and they are keyed here exactly as they are named in
        # /object_info's format spec -- VHS reads them by name, not position.
        # trim_to_audio stays False: H3 generates the pair jointly, so trimming
        # video to the audio track can only lose frames it meant to keep.
        "13": {"class_type": "VHS_VideoCombine",
               "inputs": {"images": ["11", 0], "audio": ["12", 0],
                          "frame_rate": FPS, "loop_count": 0,
                          "filename_prefix": out_prefix or f"Video/h3_{task}",
                          "format": VIDEO_FORMAT, "pix_fmt": "yuv420p",
                          "crf": 19, "save_metadata": True,
                          "trim_to_audio": False,
                          "pingpong": False, "save_output": True}},
    }

    if single_frame:
        # One frame out, so the video muxer has nothing to do and the audio
        # decoder has 0.04s of nothing to decode -- `temporal_shape(1)` gives
        # 2 audio latent steps because the streams share a clock, not because
        # there is a soundtrack. Node 12 is REMOVED rather than left dangling:
        # an unconsumed output never executes, so leaving it would be dead
        # weight in the graph that reads as an intentional wiring.
        #
        # The audio VAE loader (node 4) stays. Both the native reference node
        # and this repo's typed reference conditioner require `audio_vae`,
        # whether or not any audio is anchored.
        del g["12"]
        g["13"] = {"class_type": "SaveImage",
                   "inputs": {"images": ["11", 0],
                              "filename_prefix": out_prefix or "Image/h3_image_edit"}}

    # Resolution decides the geometry for every task except i2v, where the
    # keyframe decides it and MiniMaxH3KeyframeCanvas is the node that does.
    if task != "i2v":
        g["27"] = {"class_type": "MiniMaxH3Resolution",
                   "inputs": _resolution_widgets(cv["width"], cv["height"], length)}

    # The video VAE's ENCODE half, optionally promoted. Only the conditioning
    # node is rebound: it is the consumer that encodes references, keyframes and
    # input frames, and `VAEDecode` stays on the raw loader so the graph says
    # literally what it does -- encoder moved, decoder untouched. Both paths
    # stay correct because `MiniMaxH3VAEPrecision` normalises at the module
    # boundary rather than at the wrapper.
    #
    # NOT a shipped default, and it must not become one by drift: whether fp32
    # encode is BETTER is unmeasured, and a rendered pair cannot measure it
    # (CLAUDE.md -- the trajectory diverges completely from any numerical
    # perturbation). These arms exist to price it and to prove it runs.
    # `bench/grade_vae_encoder_precision.py` is the comparison that is
    # controlled by construction.
    vae_enc = ["3", 0]
    if vae_encoder:
        g["47"] = {"class_type": "MiniMaxH3VAEPrecision",
                   "inputs": {"vae": ["3", 0], "encoder": vae_encoder,
                              "decoder": "unchanged"}}
        vae_enc = ["47", 0]

    if ref:
        slots = _ref_image_slots(ref_images_on, ref_image_count, ref_images)
        n_refs = len(slots) + int(ref_video) + int(ref_audio)
        if not n_refs:
            raise SystemExit("r2v graph has no references to condition on")
        if n_refs > len(_REF_APPEND_NODES):
            raise SystemExit(
                f"r2v graph needs {n_refs} typed append nodes, but only "
                f"{len(_REF_APPEND_NODES)} ids are reserved")
        terminal_ref = _REF_APPEND_NODES[n_refs - 1]
        # Absent rather than null when `ref_latents` is off: an optional
        # socket the graph does not name is what the executor passes as
        # None, and a `null` literal is what it rejects.
        cond_vaes = ({"vae": vae_enc, "audio_vae": ["4", 0]}
                     if ref_latents else {})
        if native_ref and (ref_video or ref_audio):
            raise SystemExit(
                "native_ref emits core's still sockets only; wire a video or "
                "audio reference through the typed chain")
        if native_ref:
            # Core's node, sized by its own `match` rule (each still scaled
            # down to the generation's pixel area, one tensor for both the
            # VAE and Qwen). Not the typed path's sizing, and the arm notes
            # say so: this is the A/B against our conditioner, not a
            # same-footing twin of it.
            g["5"] = {"class_type": "MiniMaxH3ReferenceToVideo",
                      "inputs": {"clip": ["2", 0], **cond_vaes,
                                 "prompt": prompt,
                                 "width": ["27", 0], "height": ["27", 1],
                                 "length": ["27", 2],
                                 "ref_image_size": "match",
                                 # Autogrow sockets are dotted and count
                                 # from zero; `bench/check_reference_order.py`
                                 # drives core with the same spelling.
                                 **{f"ref_images.ref_image_{i}": [load_id, 0]
                                    for i, (load_id, _name)
                                    in enumerate(slots)}}}
        else:
            g["5"] = {"class_type": "MiniMaxH3ReferenceConditioning",
                      "inputs": {"clip": ["2", 0], **cond_vaes,
                                 "references": [terminal_ref, 0],
                                 "prompt": prompt,
                                 "width": ["27", 0], "height": ["27", 1],
                                 # Wired to Resolution so duration and geometry
                                 # continue to move together in API sweeps.
                                 "length": ["27", 2],
                                 "video_policy": ref_video_policy,
                                 "image_policy": ref_image_policy}}
        # No fit node. `MiniMaxH3AppendRefImage` carries `allow_upscale` and
        # `short_edge` itself and the conditioner performs ONE resize with the
        # canvas in scope, so the loader wires straight to the append. Before
        # that fold every reference paid a second full lanczos pass and a
        # second float32 -> uint8 -> float32 quantization, because the sizing
        # decision and its consumer were two nodes apart.
        #
        # `slots` still carries a fit id per reference and it is deliberately
        # left unallocated: reusing the loader and append ids keeps every
        # existing graph's node numbering, so the regeneration diff is the fold
        # and nothing else.
        for load_id, fname in slots:
            g[load_id] = {"class_type": "LoadImage", "inputs": {"image": fname}}
        chain = None
        append_ids = iter(_REF_APPEND_NODES)
        # Under `native_ref` the loaders feed core's sockets directly and no
        # typed chain exists, so the append loop runs over nothing.
        for load_id, _fname in ([] if native_ref else slots):
            append_id = next(append_ids)
            g[append_id] = {"class_type": "MiniMaxH3AppendRefImage",
                            "inputs": _append_image_inputs(load_id, chain, ref_upscale,
                                                           ref_qwen_short_edge)}
            chain = [append_id, 0]
        if ref_video:
            # There is NO fit node on this path, deliberately. The image path
            # has one because ComfyUI clamps reference images with
            # min(1.0, 2048/short_edge) where the reference does not. The video
            # path has the SAME class of divergence -- ComfyUI refuses to
            # upscale a reference video, the reference puts it on the full
            # canvas rule -- but closing it is expensive in a way the image one
            # is not: a 5s reference at full canvas is +32,256 rows, against
            # +7,168 for a `max` image reference. So the divergence is
            # documented and left open until the cost is known to buy anything.
            #
            # The typed append owns both media streams and the loader metadata.
            # The compiler emits its soundtrack immediately before its video,
            # preserving legacy labels while making ownership structural.
            # `start_time` 0 is the ffmpeg loader's form of the cv2 one's
            # skip_first_frames 0 / select_every_nth 1; it has no nth input.
            g["28"] = {"class_type": REF_VIDEO_LOADER,
                       "inputs": {"video": PLACEHOLDER_VIDEO,
                                  "force_rate": REF_VIDEO_FORCE_RATE,
                                  "custom_width": 0, "custom_height": 0,
                                  "frame_load_cap": length,
                                  "start_time": 0.0, "format": "AnimateDiff"}}
            g["28"]["inputs"]["video"] = (PLACEHOLDER_VIDEO if ref_video_audio
                                          else PLACEHOLDER_VIDEO_SILENT)
            append_id = next(append_ids)
            append_inputs = {"frames": ["28", 0], "video_info": ["28", 3]}
            if ref_video_audio:
                append_inputs["soundtrack"] = ["28", 2]
            if chain is not None:
                append_inputs["references"] = chain
            g[append_id] = {"class_type": "MiniMaxH3AppendRefVideo",
                            "inputs": append_inputs}
            chain = [append_id, 0]
        if ref_audio:
            # Standalone audio is appended after visual references, matching
            # the legacy conditioner's presentation order. The typed compiler
            # performs the generation-duration cap internally.
            g["33"] = {"class_type": "LoadAudio",
                       "inputs": {"audio": PLACEHOLDER_AUDIO}}
            append_id = next(append_ids)
            append_inputs = {"audio": ["33", 0]}
            if chain is not None:
                append_inputs["references"] = chain
            g[append_id] = {"class_type": "MiniMaxH3AppendRefAudio",
                            "inputs": append_inputs}
            chain = [append_id, 0]
        if not native_ref and chain != [terminal_ref, 0]:
            raise AssertionError(
                f"typed reference chain ended at {chain}, expected "
                f"{[terminal_ref, 0]}")
    else:
        # i2v takes its geometry from the keyframe node (below); every other
        # task takes it from Resolution, so the cost of the choice is visible
        # on the node where the choice is made.
        #
        # `MiniMaxH3Conditioning`, not core's `MiniMaxH3ImageToVideo`. It owns
        # the canvas itself, so the separate `MiniMaxH3KeyframeCanvas` that
        # used to sit at node 17 and hand sizes forward is gone -- one geometry
        # owner rather than two in series. Current ComfyUI owns the H3 special
        # tokens in its native tokenizer; the ignored legacy schema slot is
        # omitted from new API workflows.
        inputs = {"clip": ["2", 0], "vae": vae_enc, "prompt": prompt}
        if task == "i2v":
            # `canvas` carries what node 17's `mode` used to: derive the canvas
            # from the keyframe, or hold the geometry the caller typed. The
            # length window and the aspect refusal ride along inside the node,
            # so a graph edited in the UI afterwards keeps both.
            inputs |= {"width": cv["width"], "height": cv["height"],
                       "length": length,
                       "canvas": ("from_keyframe"
                                  if canvas_mode == "match_keyframe"
                                  else "explicit")}
            # first_frame only. Wiring `last_frame` from a second LoadImage
            # turns this into the fl2va task the checkpoint is named for; every
            # other node stays the same. Unlike core, wiring ONLY `last_frame`
            # is now also a valid graph -- the lone frame anchors the canvas
            # instead of being cropped into one chosen elsewhere.
            if first_frame:
                g["15"] = {"class_type": "LoadImage",
                           "inputs": {"image": PLACEHOLDER_IMAGE_A}}
                inputs["first_frame"] = ["15", 0]
            if last_frame:
                # The second LoadImage is the whole difference between
                # i2va and fl2va. The canvas still comes from the FIRST
                # frame under `from_keyframe` -- the release resolves it
                # on `keyframes[0]` and the closing frame cover-crops to
                # match -- so wiring this does not move the geometry
                # owner, and both placeholders are square, which keeps
                # the crop a no-op until somebody loads real stills.
                g["16"] = {"class_type": "LoadImage",
                           "inputs": {"image": PLACEHOLDER_IMAGE_B}}
                inputs["last_frame"] = ["16", 0]
        else:
            # No keyframe, so there is nothing to derive a canvas from and
            # Resolution owns it. `canvas` is inert on this path and is stated
            # rather than left to the default, so the graph says which rule it
            # is under.
            inputs |= {"width": ["27", 0], "height": ["27", 1],
                       "length": ["27", 2], "canvas": "explicit"}
        g["5"] = {"class_type": "MiniMaxH3Conditioning", "inputs": inputs}

    model_src = ["1", 0]
    if lora is not None:
        # Before the attention patches, not after. Either order renders -- a
        # LoRA patches weights and our node patches an attention function, so
        # they touch different surfaces -- but applying the LoRA clones the
        # ModelPatcher, and keeping that clone upstream of both attention
        # nodes avoids inserting it between the two that have to compose.
        # The load-bearing ordering constraint is sage-then-Sol (see
        # docs/SOLATTN.md's Ordering section). A LoRA in front of both is
        # orthogonal to it and does not belong in that constraint.
        # Node id 18; 20/21/22 are already spoken for.
        # The turbo pack's loader is not a drop-in for LoraLoaderModelOnly and
        # substituting one for the other is a silent-wrong, not an error: our
        # base is PRUNED, and this LoRA's time conditioning has to be
        # re-injected at run time from a grid the pack ships. The stock loader
        # applies the weights, skips that, and reports nothing.
        # A third loader, and it is not interchangeable with either of the
        # others. A PDD file reaches the model on three surfaces and only one
        # of them is a weight patch: the adaln update is a runtime injection
        # on our pruned base, and the per-interval output heads are not a
        # delta at all. `LoraLoaderModelOnly` would apply the 208 backbone
        # modules, skip the rest with a log line, and render -- the same
        # silent-partial shape the pack note above describes.
        if pdd:
            # `steps` here is the SAME resolved value BasicScheduler would get,
            # and on a non-split PDD graph this node's SIGMAS output replaces
            # BasicScheduler entirely (see `_sigma_src` below). The step count
            # then lives on the node that owns the 32-point grid, which is the
            # only node able to reject a count that does not tile it -- and it
            # raises rather than warning, before sampling starts.
            g["18"] = {"class_type": "MiniMaxH3PDDLoRA",
                       "inputs": {"model": model_src, "lora_name": lora[0],
                                  "strength": lora[1],
                                  # The same number as `strength`, written out
                                  # rather than left to the -1.0 sentinel that
                                  # means "follow". Identical behaviour; the
                                  # graph says what it does, and a reader does
                                  # not have to know that a negative widget is
                                  # not a negative scale.
                                  # -1.0, not lora[1] -- the sentinel for
                                  # "follow strength". See the UI builder.
                                  "head_strength": -1.0,
                                  "patch_heads": pdd_heads,
                                  "nfe": pdd_nfe,
                                  # 0 on a split graph. There, `_sigma_src`
                                  # below keeps BasicScheduler and nothing
                                  # consumes this node's SIGMAS -- but a
                                  # non-zero `steps` still reaches
                                  # `resolve_emit_steps`, which RAISES at load
                                  # on a count that does not tile the grid. A
                                  # split arm at 6 steps would be refused for
                                  # a schedule it never uses. 0 is the inert
                                  # setting the tooltip promises for this case.
                                  # Wired from a PrimitiveInt (node 61) rather
                                  # than set here, so the arm's step count is
                                  # ONE visible number in the workflow instead
                                  # of a widget buried in the loader. Owner
                                  # decision 2026-08-28. Split graphs keep the
                                  # literal 0: nothing consumes their SIGMAS,
                                  # and a non-zero value can refuse at load.
                                  # 0 under `manual_sigmas` too: ManualSigmas
                                  # replaces the schedule this node would emit,
                                  # and 0 is the one value `resolve_emit_steps`
                                  # never refuses -- which matters because a
                                  # tail-weighted partition runs 5 or 6
                                  # evaluations and neither divides the grid.
                                  "steps": (0 if (split_at or manual_sigmas)
                                            else ["61", 0]),
                                  "backbone_apply": pdd_backbone_apply}}
            if not split_at and not manual_sigmas:
                g["61"] = {"class_type": "PrimitiveInt",
                           "inputs": {"value": _resolved_steps}}
        elif lora_branch:
            # The granular inputs are written at their defaults, so a graph says
            # what it does and a runner patch can reach them (0.152.0).
            g["18"] = {"class_type": LORA_BRANCH_NODE,
                       "inputs": {"model": model_src, "lora_name": lora[0],
                                  "strength": lora[1], "modules": "all", "blocks": lora_blocks,
                                  "start_percent": 0.0, "end_percent": 1.0}}
        else:
            g["18"] = {"class_type": "LoraLoaderModelOnly",
                       "inputs": {"model": model_src, "lora_name": lora[0],
                                  "strength_model": lora[1]}}
        model_src = ["18", 0]
    # At the base checkpoint's own 12/3, so it changes nothing by default. It
    # is here to be edited: the turbo LoRAs carry their own training shifts
    # (the 768p 4-step wants 6/3), and a graph without this node gives you
    # nowhere to set that and no hint you needed to. Upstream of sage so the
    # sage-then-Sol adjacency below stays intact -- this patches model
    # sampling, which is a different surface from either of them.
    #
    # OMITTED FROM PDD GRAPHS at the default shift, on the owner's call
    # 2026-08-31, because there it is a no-op that reads as a knob. The PDD
    # node emits SIGMAS from the shift its file was fused at, so the schedule
    # never comes off `model_sampling` here; and every surface this node
    # touches already carries 12/3 without it -- `ModelSamplingAV + CONST` is
    # what `ModelType.FLOW_AV` selects anyway (`comfy/model_base.py`), the
    # values match `MiniMaxH3.sampling_settings` (`comfy/supported_models.py`),
    # and the DiT falls back to its own `sigma_shift_video/audio` ctor defaults
    # when the `transformer_options` keys are absent (`comfy/ldm/minimax/
    # model.py`). `pdd_lora.py::check_shift` covers the absent case explicitly:
    # no key means it compares against the model's class default instead.
    # So it patched the model into what it already was, while inviting an edit
    # that `check_shift` raises on at step 0.
    #
    # The condition is `sh == SIGMA_SHIFT`, not `not pdd`, so a PDD arm fused
    # at some other shift gets the node back rather than silently losing the
    # only place to set it.
    # Node id 19; 18 is the LoRA and 20/21/22 are already spoken for.
    if channel_balance is not None:
        # Weight patches on q_norm/k_norm of the lopsided blocks; no runtime
        # cost, so its position only needs to precede the attention nodes.
        # Node id 50: 47-49 are taken by the VAE-precision and freeze arms.
        g["50"] = {"class_type": "MiniMaxH3ChannelBalance",
                   "inputs": {"model": model_src, "balance": channel_balance,
                              "blocks": "49", "alpha": 0.5, "loud_share": 0.15}}
        model_src = ["50", 0]
    sh = shift if shift is not None else SIGMA_SHIFT
    if not (pdd and sh == SIGMA_SHIFT):
        g["19"] = {"class_type": "MiniMaxH3SigmaShift",
                   "inputs": {"model": model_src, **sh}}
        model_src = ["19", 0]
    if dense_backend is not None:
        # Core's node: `set_model_optimized_attention`, which is the function
        # our Sol node's dense fallback calls, so Sol composes on top of it
        # exactly as core's own block-sparse node would. Node id 58. It was 60
        # for the three probes that carried it on 2026-09-15, which is also
        # ManualSigmas' id below: harmless while no probe set both, a silent
        # overwrite once every graph carried the backend. 50-57 are the typed
        # reference appends (`_REF_APPEND_NODES`).
        g["58"] = {"class_type": "ModelAttentionBackend",
                   "inputs": {"model": model_src, "attention": dense_backend}}
        model_src = ["58", 0]
    if sage:
        g["20"] = {"class_type": "MiniMaxH3SageAttention",
                   "inputs": {"model": model_src, **dict(
                       SAGE_NODE,
                       **({} if head_chunks is None
                          else {"head_chunks": head_chunks}),
                       **({} if sage_mode is None else {"mode": sage_mode}))}}
        model_src = ["20", 0]
    if core_vsa is not None:
        if sol is not None:
            raise SystemExit("core_vsa takes Sol's slot and replaces the block attention; "
                             "pass it with sol=None")
        g["21"] = {"class_type": SOL_CORE_NODE,
                   "inputs": {"model": model_src, **core_vsa}}
        model_src = ["21", 0]
    if sol is not None and sol_impl == "core":
        # ComfyUI core's own Sol node in our node's slot, at ITS OWN schema
        # defaults (h3_config.SOL_CORE_DEFAULTS, inherited and re-read against
        # /object_info on every validating build). Also after sage: core puts
        # its override on top of whatever override is already on the hook and
        # sends every call it declines to that one, so sage stays the floor.
        # The `sol` recipe is ignored here on purpose -- this arm exists to
        # render core's node as core ships it.
        g["21"] = {"class_type": SOL_CORE_NODE,
                   "inputs": {"model": model_src, **SOL_CORE_DEFAULTS}}
        model_src = ["21", 0]
    elif sol is not None:
        if sol_impl != "ours":
            raise SystemExit(f"sol_impl={sol_impl!r} is not 'ours' or 'core'")
        # After sage, never before -- SolAttn composes with the attention
        # patches it finds, and reversed it overwrites ours and you silently
        # get sage only. Node id 21 matches `bench/bench_e2e_h3.py`.
        g["21"] = {"class_type": SOL_NODE,
                   "inputs": {"model": model_src, **sol_api_inputs(sol)}}
        model_src = ["21", 0]
    # `SageChainAssert` (node 23, and 43 on the split chain) stood here until
    # 2026-09-17, when the owner took it out of every generated graph. On the
    # default chain it could only confirm that no sage kernel ran, and its
    # flags, fixed at generation time, went stale the moment a graph was
    # edited in the editor. `bench/check_attention_defaults.py` grades the
    # wiring, and Sol logs its own composition. The node was deleted in 0.173.0
    # (owner, 2026-09-29); ids 23 and 43 stay reserved.
    if exact_blocks is not None:
        # After the attention nodes; exact blocks wrap on top (the node's
        # forward survives either order).
        g["51"] = {"class_type": "MiniMaxH3ExactBlocks",
                   "inputs": {"model": model_src, "blocks": exact_blocks}}
        model_src = ["51", 0]

    if cache is not None:
        # Step caching, after the attention nodes: the cache is a
        # forward-skipping wrapper on top of the attention composition, not
        # part of it. On a reused step nothing downstream of the
        # wrapper runs -- sage and Sol included -- which is the mechanism, not
        # a conflict. See CACHE_NODE in h3_config.py for why this arm exists
        # and its re-noising-sampler caveat. Node id 44: 28-33 are the reference loaders
        # and split path, 34-39 the reference image slots, 40-43 the plain
        # chain `split_at` builds.
        g["44"] = {"class_type": CACHE_NODE_CLASS,
                   "inputs": {"model": model_src, **cache}}
        model_src = ["44", 0]

    # Reports what the assembled conditioning actually costs, before the
    # sampler runs. Pass-through, so it cannot change the render.
    g["26"] = {"class_type": "MiniMaxH3Preflight",
               "inputs": {"conditioning": ["5", 0], "samples": ["5", 1]}}

    # Where the sampler's sigmas come from, and it is the whole point of the
    # PDD rewiring.
    #
    # Every knob that can put a distilled render off its own grid --
    # `scheduler`, `steps` -- lives on BasicScheduler, which sits DOWNSTREAM of
    # every model-patch node. So the PDD node could only ever observe the
    # schedule after the fact and report on it, and three separate footguns
    # followed from that: a scheduler that is not `simple`, a step count that
    # does not tile the 32-point grid, and evaluation off the block boundaries
    # entirely. Each was caught, if at all, by a static check over SHIPPED
    # graphs -- so a hand-edited or hand-built graph had nothing.
    #
    # Emitting the schedule from the PDD node inverts the dependency. The
    # sampler steps at exactly the boundaries the heads were fused for, there
    # is no scheduler widget to get wrong, and off-grid is not expressible.
    # `ComfyUI-UtilsCollection` reached the same design from the other end --
    # its off-grid error tells you to use its SIGMAS output; this makes that
    # the wiring rather than the advice.
    #
    # Numerically inert on every shipped PDD graph: the node emits
    # `1 - pdd_time_grid`, which IS the plain shifted schedule for the block
    # count, and that is bit-identical to `BasicScheduler(simple, N)` at 2, 4
    # and 8 steps. Graded by `bench/check_pdd_sigmas.py` against ComfyUI's own
    # `calculate_sigmas` rather than against a value computed here.
    #
    # A split graph keeps BasicScheduler: `SplitSigmas` wants one schedule fed
    # to both halves and that combination has never shipped with PDD, so it
    # keeps the old path rather than inheriting an untested one.
    if manual_sigmas:
        # An explicit non-uniform partition. The node's SIGMAS output can only
        # express counts that DIVIDE the 32-point grid, so a tail-weighted 5- or
        # 6-evaluation schedule is unreachable through it today.
        g["60"] = {"class_type": "ManualSigmas",
                   "inputs": {"sigmas": manual_sigmas}}
    _sigma_src = (["60", 0] if manual_sigmas
                  else ["18", 1] if (pdd and lora is not None and not split_at)
                  else ["8", 0])
    g["10"]["inputs"]["sigmas"] = _sigma_src
    if _sigma_src == ["8", 0]:
        g["8"]["inputs"]["model"] = model_src
    else:
        del g["8"]

    # The guider, from the same variable the sigma source above used.
    g["9"]["inputs"]["model"] = model_src
    g["9"]["inputs"]["conditioning"] = ["26", 0]
    g["10"]["inputs"]["latent_image"] = ["26", 1]

    if split_at:
        # Two-stage split. ONE BasicScheduler feeds SplitSigmas, so both halves
        # sample the same curve -- that shared schedule is the whole
        # precondition, and it is why both stages must also share a shift.
        #
        # Built on SamplerCustomAdvanced rather than KSamplerAdvanced. Krea 2's
        # version of this uses KSamplerAdvanced, and KSamplerAdvanced with
        # add_noise disabled was BROKEN on nested latents until core 27bca654
        # (2026-08-12): it called torch.zeros(latent.size()) on a NestedTensor,
        # which is what H3's AV latent is. The custom-sampler route was never
        # broken.
        #
        # `split_at` counts steps of the shared schedule, so at 8 steps
        # split_at=1 means stage 1 runs step 0 alone. H3's schedule is far more
        # front-loaded than Krea 2's -- at shift 12 seven of eight evals sit at
        # sigma >= 0.8 and the final interval covers the bottom 63% of the
        # range -- so the useful boundary is much lower here. Sweep from 1.
        if not lora:
            raise SystemExit(
                "split_at needs a `lora`: the point of the split is that the "
                "two stages run different models. Without one both halves are "
                "the same model and the split is an expensive no-op.")
        g["29"] = {"class_type": "SplitSigmas",
                   "inputs": {"sigmas": ["8", 0], "step": split_at}}
        g["30"] = {"class_type": "DisableNoise", "inputs": {}}

        # `model_src` carries the LoRA. The second chain is the plain model.
        plain_src = _plain_model_chain(g, sage=sage, sol=sol, shift=shift,
                                       head_chunks=head_chunks,
                                       dense_backend=dense_backend)
        # base_last: distilled student takes the high-noise majority, the plain
        #   base model finishes. This is the ordering for ref2v -- the
        #   student's measured deficit is high-frequency detail, resolved at
        #   low sigma, and high-frequency identity is what a reference is for,
        #   so the intuitive ordering puts its weakness where demand is highest.
        # base_first: the Krea 2 ordering, base for composition then a fast
        #   distilled finish. Right when the finish is about sharpness.
        stage1, stage2 = ((model_src, plain_src) if split_base_last
                          else (plain_src, model_src))
        g["8"]["inputs"]["model"] = stage1
        g["9"]["inputs"]["model"] = stage1
        g["31"] = {"class_type": "BasicGuider",
                   "inputs": {"model": stage2, "conditioning": ["26", 0]}}
        g["32"] = {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["30", 0], "guider": ["31", 0],
                              "sampler": ["7", 0], "sigmas": ["29", 1],
                              "latent_image": ["10", 0]}}
        # Stage 1 takes the high half and hands its leftover-noise latent on.
        # `add_noise` is not a knob here: stage 2's noise source is
        # DisableNoise, which is the custom-sampler spelling of it.
        g["10"]["inputs"]["sigmas"] = ["29", 0]
        g["11"]["inputs"]["samples"] = ["32", 0]
        g["12"]["inputs"]["samples"] = ["32", 0]
        if stamp:
            raise SystemExit("stamp and split_at are not wired together")

    if stamp:
        # Bench only. Sits inline between the sampler and both decoders so it
        # has a real data dependency on the sampler's output -- ComfyUI orders
        # by dependency, not graph position, and a stamp with no such edge can
        # legally run BEFORE sampling and record pre-render state. It also
        # needs SIGMAS: n_sparse is the sigma window intersected with the
        # schedule and is readable from nothing else.
        g["22"] = {"class_type": "MiniMaxH3ProvenanceStamp",
                   "inputs": {"latent": ["10", 0], "model": model_src,
                              "sigmas": _sigma_src, "note": f"bench {task}"}}
        g["11"]["inputs"]["samples"] = ["22", 0]
        g["12"]["inputs"]["samples"] = ["22", 0]

    if freeze_audio:
        if single_frame or split_at or stamp:
            raise SystemExit(
                "freeze_audio does not compose with single_frame, split_at or "
                "stamp: each rewires the sampler's latent or the audio decoder")
        # Node ids 48 and 49 were free (45 too). The node takes the preflight's
        # latent so the report still describes the graph that runs, and hands
        # the sampler the same latent with the track written in and the mask
        # attached. Node 12 goes: nothing may consume the decoded audio stream
        # on this graph, and an unconsumed node reads as intentional wiring.
        g["48"] = {"class_type": "LoadAudio", "inputs": {"audio": freeze_track}}
        g["49"] = {"class_type": "MiniMaxH3FreezeAudio",
                   "inputs": {"latent": ["26", 1], "audio_vae": ["4", 0],
                              "audio": ["48", 0],
                              "start_seconds": freeze_start,
                              "audio_mask": freeze_mask, "level": "clip_guard"}}
        g["10"]["inputs"]["latent_image"] = ["49", 0]
        g["13"]["inputs"]["audio"] = ["49", 1]
        del g["12"]
        if freeze_guide:
            # Between the conditioner and the preflight on the positive path,
            # so the preflight prices the rows the guide adds. Node id 70.
            g["70"] = {"class_type": "MiniMaxH3AddGuide",
                       "inputs": {"positive": ["5", 0], "audio_vae": ["4", 0],
                                  "latent": ["5", 1], "audio": ["48", 0],
                                  "frame_idx": 0}}
            g["26"]["inputs"]["conditioning"] = ["70", 0]
    elif freeze_guide:
        raise SystemExit("freeze_guide needs freeze_audio")

    if freeze_windows:
        if freeze_audio or single_frame or split_at or stamp:
            raise SystemExit("freeze_windows composes with none of freeze_audio, single_frame, split_at, stamp")
        if freeze_windows != 2:
            raise SystemExit("freeze_windows: only the two-window seam is built here; a loop needs TensorLoop")
        # Window 0: the window node in the freeze node's place. Window 1: a
        # second window node fed the first sampler's output, its own noise
        # (seed + 1), its own guider and sampler on the same model, sigmas
        # and conditioning; decoded separately, its context frames dropped,
        # the two image batches joined, and the muxer on the span audio.
        # Node ids 48 (LoadAudio), 62-69 (free).
        g["48"] = {"class_type": "LoadAudio", "inputs": {"audio": freeze_track}}
        g["62"] = {"class_type": "MiniMaxH3FreezeAudioWindow",
                   "inputs": {"latent": ["26", 1], "audio_vae": ["4", 0], "audio": ["48", 0],
                              "start_seconds": 0.0, "context_frames": freeze_context,
                              "audio_mask": freeze_mask, "level": "clip_guard"}}
        g["10"]["inputs"]["latent_image"] = ["62", 0]
        g["63"] = {"class_type": "MiniMaxH3FreezeAudioWindow",
                   "inputs": {"latent": ["26", 1], "audio_vae": ["4", 0], "audio": ["48", 0],
                              # the first window's next_start_seconds, wired
                              "start_seconds": ["62", 4], "context_frames": freeze_context,
                              "previous": ["10", 0],
                              "audio_mask": freeze_mask, "level": "clip_guard"}}
        g["64"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + 1}}
        g["65"] = {"class_type": "BasicGuider",
                   "inputs": {"model": g["9"]["inputs"]["model"], "conditioning": ["26", 0]}}
        g["66"] = {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["64", 0], "guider": ["65", 0], "sampler": ["7", 0],
                              "sigmas": g["10"]["inputs"]["sigmas"], "latent_image": ["63", 0]}}
        g["67"] = {"class_type": "VAEDecode", "inputs": {"samples": ["66", 0], "vae": ["3", 0]}}
        g["68"] = {"class_type": "ImageFromBatch",
                   "inputs": {"image": ["67", 0], "batch_index": freeze_context,
                              "length": length - freeze_context}}
        g["69"] = {"class_type": "ImageBatch", "inputs": {"image1": ["11", 0], "image2": ["68", 0]}}
        g["13"]["inputs"]["images"] = ["69", 0]
        g["13"]["inputs"]["audio"] = ["63", 2]
        del g["12"]

    if freeze_gain:
        # Node id 73, between the model chain and the guider (and the
        # scheduler, which reads the same model object).
        g["73"] = {"class_type": "MiniMaxH3AudioAttentionGain",
                   "inputs": {"model": g["9"]["inputs"]["model"], "key_gain": 1.0, "value_gain": 1.0,
                              "start_percent": 0.0, "end_percent": 1.0, "blocks": "all",
                              "include_reference_rows": False}}
        g["9"]["inputs"]["model"] = ["73", 0]

    if freeze_song:
        if freeze_audio or freeze_windows or single_frame or split_at or stamp or ref or task == "i2v":
            raise SystemExit("freeze_song is a t2v chain and composes with no other latent-side knob")
        _sig = g["10"]["inputs"]["sigmas"]
        _model = g["9"]["inputs"]["model"]
        for nid in ("5", "6", "9", "10", "11", "12", "13", "26"):
            g.pop(nid, None)
        g["48"] = {"class_type": "LoadAudio", "inputs": {"audio": freeze_track}}
        if freeze_song_source and freeze_song_seconds is None:
            raise SystemExit("freeze_song_source needs freeze_song_seconds: the loader holds every frame "
                             "it loads, so the graph says how much of the video it takes")
        g["74"] = {"class_type": "MiniMaxH3AudioFreezeSong",
                   "inputs": {"model": _model, "clip": ["2", 0], "vae": vae_enc, "audio_vae": ["4", 0],
                              "audio": ["48", 0], "sampler": ["7", 0], "sigmas": _sig,
                              "prompt": prompt, "width": cv["width"], "height": cv["height"],
                              "window_frames": length, "context_frames": freeze_context,
                              # `extent` is a DynamicCombo (2026-09-13): dotted
                              # member in the API form, as `size_policy` is.
                              # `freeze_song_seconds=None` is the whole track.
                              **({"extent": "first_seconds", "extent.seconds": freeze_song_seconds}
                                 if freeze_song_seconds is not None else {"extent": "whole"}),
                              "seed": seed,
                              "audio_mask": freeze_mask, "level": "clip_guard",
                              "filename_prefix": out_prefix or "Video/h3_song", "crf": 19,
                              "timeline": freeze_song_timeline, "preview": False,
                              "save_metadata_png": True, "keep_windows": True,
                              "reuse_windows": True}}
        # The node's report on a Preview as Text, so the plan a `preview` run
        # prints shows on the canvas in the editor.
        g["75"] = {"class_type": "PreviewAny", "inputs": {"source": ["74", 1]}}
        if freeze_song_refs:
            # The song node compiles its references itself, once per distinct
            # prompt, so only the conditioner differs from a reference graph:
            # the loaders and appends are theirs, same ids and same inputs.
            chain = None
            append_ids = iter(_REF_APPEND_NODES)
            for load_id, fname in _ref_image_slots(True, len(freeze_song_refs), freeze_song_refs):
                g[load_id] = {"class_type": "LoadImage", "inputs": {"image": fname}}
                append_id = next(append_ids)
                g[append_id] = {"class_type": "MiniMaxH3AppendRefImage",
                                "inputs": _append_image_inputs(load_id, chain, ref_upscale,
                                                               ref_qwen_short_edge)}
                chain = [append_id, 0]
            g["74"]["inputs"]["references"] = chain
        if freeze_song_lists:
            # typed lists, chained in order into the song node's `lists`
            if len(freeze_song_lists) > len(_PROMPT_LIST_NODES):
                raise SystemExit(f"at most {len(_PROMPT_LIST_NODES)} prompt lists")
            chain = None
            for list_id, (name, values, order, shuffle) in zip(_PROMPT_LIST_NODES, freeze_song_lists):
                g[list_id] = {"class_type": "MiniMaxH3PromptList",
                              "inputs": {"name": name, "source": "typed", "source.values": values,
                                         "order": order, "shuffle": shuffle,
                                         **({"lists": chain} if chain is not None else {})}}
                chain = [list_id, 0]
            g["74"]["inputs"]["lists"] = chain
        if freeze_song_source:
            # The video is the track: its frames at 24 fps and canvas width
            # (the loader holds them all, so not at the file's own size), its
            # audio in place of LoadAudio. Ids 100-105, used by nothing else.
            # The cap is the count the song node's plan reads for this extent, no more:
            # until 2026-10-06 it was the extent plus a whole window, and the tracker
            # worked on every frame of that (`bench/results/2026-10-06_masked_render_time_breakdown.md`).
            g.pop("48")
            g["28"] = {"class_type": REF_VIDEO_LOADER,
                       "inputs": {"video": PLACEHOLDER_VIDEO, "force_rate": REF_VIDEO_FORCE_RATE,
                                  "custom_width": cv["width"], "custom_height": 0,
                                  "frame_load_cap": _song_frames_read(freeze_song_seconds, length, freeze_context,
                                                                      freeze_song_timeline),
                                  "start_time": 0.0, "format": "AnimateDiff"}}
            g["74"]["inputs"]["audio"] = ["28", 2]
            g["100"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SEGMENTER}}
            # Subject, then what happens to them: two nodes on purpose (owner,
            # 2026-10-04), so a change to the margin or the composite never
            # runs the tracker again. Its `preview` and `report` outputs stay
            # unwired: a consumer would run it on every queue and defeat the
            # kept mask. Id 105; 101-103 were core's tracker chain until then.
            g["105"] = {"class_type": "MiniMaxH3SubjectTrack",
                        "inputs": {"frames": ["28", 0], "segmenter": ["100", 0], "segmenter_clip": ["100", 1],
                                   **SUBJECT_TRACK}}
            g["104"] = {"class_type": "MiniMaxH3MaskedSource",
                        # the segmenter is wired whether or not `replace` reads it, so
                        # changing that one choice needs no rewiring
                        # `shot_table` is the tracker's fourth output: lazy with the mask and
                        # kept with it, so it does not run the tracker on a hit either, and the
                        # song node writes it beside the video (`shot_table.py`, 2026-10-05).
                        "inputs": {"frames": ["28", 0], "mask": ["105", 0], **(masked_source or MASKED_SOURCE),
                                   "segmenter": ["100", 0], "segmenter_clip": ["100", 1],
                                   "shot_table": ["105", 3]}}
            g["74"]["inputs"]["source"] = ["104", 0]
            # the render stacked over what was regenerated, beside the render (2026-10-06, at the
            # owner's ask). Written only here: the input does nothing without a source.
            g["74"]["inputs"]["save_mask_review"] = True
            if masked_parts:
                # ids 107 and 108. The part node runs on the tracker's mask and
                # only when the Masked Source asks for `parts`, which it does
                # not when a kept mask matches.
                if g["104"]["inputs"]["replace"] != "the wired parts":
                    raise SystemExit("masked_parts needs a masked_source whose replace is `the wired parts`")
                g["107"] = {"class_type": "MiniMaxH3Sapiens2Loader", "inputs": dict(SAPIENS2)}
                g["108"] = {"class_type": "MiniMaxH3SubjectParts",
                            "inputs": {"sapiens2": ["107", 0], "frames": ["28", 0], "subject_mask": ["105", 0],
                                       **SUBJECT_PARTS}}
                g["104"]["inputs"]["parts"] = ["108", 0]
            if masked_review:
                # ids 109 and 110, in a review graph only: a consumer of the
                # tracker's tiles runs the tracker on every queue.
                g["74"]["inputs"]["preview"] = True
                g["109"] = {"class_type": "MiniMaxH3SaveShotTable",
                            "inputs": {"shot_table": ["105", 3], "preview": ["105", 1],
                                       "filename_prefix": (out_prefix or "Video/h3_song") + "_shots"}}
                g["110"] = {"class_type": "PreviewImage", "inputs": {"images": ["104", 2]}}
            if masked_prompt is not None:
                # The prompt is written by the node from what the Masked Source
                # does, and shown on it; id 106.
                g["106"] = {"class_type": MASKED_PROMPT_NODE,
                            "inputs": {**masked_prompt, "source": ["104", 0]}}
                g["74"]["inputs"]["prompt"] = ["106", 0]
                written = _prompts.carriers(g)
                written = [c.text for c in written if c.node_id == "74"]
                if written != [prompt]:
                    raise SystemExit("masked_prompt: the Masked Prompt node would not write this graph's bank "
                                     "text; rebuild the bank copy with bench/check_masked_prompt.py --write")
        elif masked_prompt is not None:
            raise SystemExit("masked_prompt needs freeze_song_source")
    elif freeze_song_refs or freeze_song_lists or freeze_song_source:
        raise SystemExit("freeze_song_refs, freeze_song_lists and freeze_song_source need freeze_song")

    if refine_cache and not audio_refine:
        raise SystemExit("refine_cache needs audio_refine")
    if audio_refine:
        # ComfyUI-H3-AudioRefine's design (coderef/ComfyUI-H3-AudioRefine,
        # MIT): pass 1 runs the distill, pass 2 runs a few undistilled steps on
        # the audio only. The refine model starts at the UNETLoader, BEFORE
        # the LoRA, and gets its own copy of the base chain -- base shift,
        # the dense backend node if the graph has one, and Sol at the base
        # recipe -- so every sampling path carries the attention the graph
        # declares. Node ids 80-87.
        if lora is None or freeze_audio or freeze_windows or freeze_song or split_at or single_frame:
            raise SystemExit("audio_refine needs a distill LoRA and composes with none of "
                             "freeze_audio, freeze_windows, freeze_song, split_at, single_frame")
        rsrc = ["1", 0]
        g["80"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": rsrc, **SIGMA_SHIFT}}
        rsrc = ["80", 0]
        if "58" in g:
            g["81"] = {"class_type": "ModelAttentionBackend",
                       "inputs": {"model": rsrc, "attention": g["58"]["inputs"]["attention"]}}
            rsrc = ["81", 0]
        if sol is not None:
            g["82"] = {"class_type": SOL_NODE,
                       "inputs": {"model": rsrc,
                                  **sol_api_inputs(sol_for_graph(False, SAMPLING["steps"]))}}
            rsrc = ["82", 0]
        if refine_cache:
            g["88"] = {"class_type": FROZEN_VIDEO_CACHE_NODE,
                       "inputs": {"model": rsrc, **FROZEN_VIDEO_CACHE}}
            rsrc = ["88", 0]
        g["83"] = {"class_type": "MiniMaxH3AudioRefineMask",
                   "inputs": {"latent": ["10", 0],
                              "video_mask": AUDIO_REFINE["video_mask"],
                              "audio_mask": AUDIO_REFINE["audio_mask"]}}
        g["84"] = {"class_type": "BasicScheduler",
                   "inputs": {"model": rsrc, "scheduler": AUDIO_REFINE["scheduler"],
                              "steps": AUDIO_REFINE["steps"], "denoise": AUDIO_REFINE["denoise"]}}
        g["85"] = {"class_type": "KSamplerSelect",
                   "inputs": {"sampler_name": AUDIO_REFINE["sampler"]}}
        g["86"] = {"class_type": "BasicGuider",
                   "inputs": {"model": rsrc, "conditioning": ["26", 0]}}
        g["87"] = {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["6", 0], "guider": ["86", 0], "sampler": ["85", 0],
                              "sigmas": ["84", 0], "latent_image": ["83", 0]}}
        g["11"]["inputs"]["samples"] = ["87", 0]
        g["12"]["inputs"]["samples"] = ["87", 0]

    if step_switch:
        if step_switch_to not in ("pdd8", "flashgen", "base", "fasth3"):
            raise SystemExit(f"step_switch_to must be pdd8, flashgen, base or fasth3, not {step_switch_to!r}")
        if (lora is None or audio_refine or split_at or freeze_audio or freeze_windows or single_frame
                or pdd != (step_switch_to in ("flashgen", "base", "fasth3"))):
            raise SystemExit("step_switch runs this graph's own chain as pass 1 (a non-PDD LoRA "
                             "before pdd8, PDD before flashgen or base) and composes with none of "
                             "audio_refine, split_at, freeze_*, single_frame")
        # Pass 2's model chain is the other distill's own graph, built by this
        # function and copied in at +200, so it carries exactly what that
        # graph carries (the LoRA node, its shift, the dense backend, Sol at
        # its recipe). The UNET loader is shared, and asserted identical.
        # Only on 0.154.8 or later: two branch LoRAs in one graph is the
        # shape the 2026-09-26 stacking bug hit (2faed1a8).
        if step_switch_to == "pdd8":
            pass2_sigmas = step_switch_sigmas or STEP_SWITCH_PASS2_SIGMAS
            g2 = build_api(task, sage=sage, prompt=prompt, length=length, seed=seed,
                           sol=(sol_for_graph(True, PDD_STEPS) if sol is not None else None),
                           sol_impl=sol_impl, dense_backend=dense_backend,
                           pdd=True, lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
                           sampler_name="euler", manual_sigmas=pass2_sigmas,
                           unet=unet, clip=clip, **canvas)
        elif step_switch_to == "base":
            if not step_switch_sigmas:
                raise SystemExit("step_switch_to='base' needs step_switch_sigmas (h3_config.STEP_SWITCH_BASE)")
            pass2_sigmas = step_switch_sigmas
            base_evals = len(pass2_sigmas.split(",")) - 1
            g2 = build_api(task, sage=sage, prompt=prompt, length=length, seed=seed,
                           sol=(sol_for_graph(False, base_evals) if sol is not None else None),
                           sol_impl=sol_impl, dense_backend=dense_backend,
                           steps=base_evals, sampler_name="euler",
                           manual_sigmas=pass2_sigmas, unet=unet, clip=clip, **canvas)
        elif step_switch_to == "fasth3":
            if not (step_switch_sigmas and step_switch_shift):
                raise SystemExit("step_switch_to='fasth3' needs step_switch_sigmas and step_switch_shift (h3_config.STEP_SWITCH_FASTH3)")
            pass2_sigmas = step_switch_sigmas
            g2 = build_api(task, sage=False, prompt=prompt, length=length, seed=seed,
                           sol=None, dense_backend=dense_backend, core_vsa=FASTH3_CONTRACT_VSA,
                           steps=len(pass2_sigmas.split(",")) - 1, sampler_name=FASTH3_CONTRACT_SAMPLER,
                           manual_sigmas=pass2_sigmas, shift=step_switch_shift,
                           unet=MODELS["unet_fasth3_v2"], clip=clip, **canvas)
        else:
            if not step_switch_sigmas:
                raise SystemExit("step_switch_to='flashgen' needs step_switch_sigmas (h3_config.STEP_SWITCH_REV)")
            pass2_sigmas = step_switch_sigmas
            g2 = build_api(task, sage=sage, prompt=prompt, length=length, seed=seed,
                           sol=(sol_for_graph(False, FLASHGEN_STEPS, distill_lora=True)
                                if sol is not None else None),
                           sol_impl=sol_impl, dense_backend=dense_backend,
                           lora=(FLASHGEN_R64_REF2VA_LORA if task == "r2v" else FLASHGEN_R64_LORA,
                                 FLASHGEN_STRENGTH), lora_branch=True,
                           lora_blocks=step_switch_blocks,
                           steps=FLASHGEN_STEPS, sampler_name=FLASHGEN_SAMPLER,
                           manual_sigmas=pass2_sigmas, unet=unet, clip=clip, **canvas)
        if (g2["1"] != g["1"]) != (step_switch_to == "fasth3"):
            raise SystemExit("step_switch: pass 1 and pass 2 load different UNETs only for fasth3")
        chain, ref = {}, g2["9"]["inputs"]["model"]
        while ref[0] != "1":
            node = g2[ref[0]]
            chain[ref[0]] = node
            ref = next(v for k, v in node["inputs"].items()
                       if isinstance(v, list) and len(v) == 2 and k == "model")
        if step_switch_to == "fasth3":
            chain["1"] = g2["1"]

        def _moved(v):
            return ([str(int(v[0]) + 200), v[1]] if isinstance(v, list) and len(v) == 2
                    and isinstance(v[0], str) and v[0] in chain else v)
        for nid, node in chain.items():
            g[str(int(nid) + 200)] = {"class_type": node["class_type"],
                                      "inputs": {k: _moved(v) for k, v in node["inputs"].items()}}
        g["120"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": pass2_sigmas}}
        g["121"] = {"class_type": "DisableNoise", "inputs": {}}
        g["122"] = {"class_type": "BasicGuider",
                    "inputs": {"model": _moved(g2["9"]["inputs"]["model"]), "conditioning": ["26", 0]}}
        g["123"] = {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["121", 0], "guider": ["122", 0], "sampler": ["7", 0],
                               "sigmas": ["120", 0], "latent_image": ["10", 0]}}
        g["11"]["inputs"]["samples"] = ["123", 0]
        g["12"]["inputs"]["samples"] = ["123", 0]

    def _save_av(src, tag, ids):
        split, vid, aud = ids
        prefix = g["13"]["inputs"]["filename_prefix"].removeprefix("Video/")
        g[split] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": src}}
        g[vid] = {"class_type": "SaveLatent",
                  "inputs": {"samples": [split, 0], "filename_prefix": f"latents/{prefix}{tag}_video"}}
        g[aud] = {"class_type": "SaveLatent",
                  "inputs": {"samples": [split, 1], "filename_prefix": f"latents/{prefix}{tag}_audio"}}

    if probe_step_x0:
        src = g["9"]["inputs"]["model"]
        prefix = g["13"]["inputs"]["filename_prefix"].removeprefix("Video/")
        g["113"] = {"class_type": "MiniMaxH3StepX0Observer",
                    "inputs": {"model": src, "filename_prefix": f"latents/{prefix}_x0",
                               "save_audio": False}}
        g["9"]["inputs"]["model"] = ["113", 0]
        if "8" in g and g["8"]["inputs"].get("model") == src:
            g["8"]["inputs"]["model"] = ["113", 0]

    if probe_core_sparse_capture:
        src = g["9"]["inputs"]["model"]
        if g.get(src[0], {}).get("class_type") != "BlockSparseAttention":
            raise SystemExit("probe_core_sparse_capture wraps core's BlockSparseAttention; "
                             f"the guider's model comes from {g.get(src[0], {}).get('class_type')}")
        g["114"] = {"class_type": "MiniMaxH3CoreSparseCapture", "inputs": {"model": src}}
        g["9"]["inputs"]["model"] = ["114", 0]
        if "8" in g and g["8"]["inputs"].get("model") == src:
            g["8"]["inputs"]["model"] = ["114", 0]

    if probe_frozen_rows:
        if not audio_refine:
            raise SystemExit("probe_frozen_rows needs audio_refine: it observes the refine pass")
        g["106"] = {"class_type": "MiniMaxH3DenoiseMaskProbe",
                    "inputs": {"model": g["86"]["inputs"]["model"]}}
        g["86"]["inputs"]["model"] = ["106", 0]
        # The scheduler reads the same model as the guider, which the refine
        # validator requires; the probe only observes, so the sigmas are unchanged.
        g["84"]["inputs"]["model"] = ["106", 0]
        _save_av(["10", 0], "_pass1", ("107", "108", "109"))
        _save_av(["87", 1], "_refine_denoised", ("110", "111", "112"))
    if step_switch and save_latents:
        _save_av(["10", 0], "_pass1", ("107", "108", "109"))

    if draft_decode:
        # Last, so it saves whatever latent the decoders read after every
        # other option has rewired them (the refine pass moves them to 87).
        # SaveLatent writes `samples` as one tensor and H3's latent is a
        # nested video/audio pair, so core's AV splitter goes first. The
        # prefixes share the clip's, so a draft's latents sort beside it.
        if single_frame or "12" not in g:
            raise SystemExit("draft_decode needs a clip with both decoders (nodes 11 and 12)")
        g["100"] = {"class_type": "VAELoader", "inputs": {"vae_name": DRAFT_VAE}}
        g["11"]["inputs"]["vae"] = ["100", 0]
    if draft_decode or save_latents:
        if single_frame or "12" not in g:
            raise SystemExit("save_latents needs a clip with both decoders (nodes 11 and 12)")
        _save_av(g["11"]["inputs"]["samples"], "", ("101", "102", "103"))

    return g


def build_decode_saved_latent(video_latent: str, audio_latent: str, *,
                              out_prefix: str = "Video/h3_decode_saved_latent") -> dict:
    """The real decode of a draft's saved latents, with no sampling.

    `LoadLatent` resolves an annotated path, so `latents/<name>.latent [output]`
    reads the file where `SaveLatent` wrote it; nothing is copied to `input/`.
    The video goes through `MODELS["video_vae"]`, the decoder every shipped
    graph uses, and the audio through the audio VAE as the draft did. Node ids
    follow `build_api` where the role is the same.
    """
    return {
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["video_vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["audio_vae"]}},
        "104": {"class_type": "LoadLatent", "inputs": {"latent": video_latent}},
        "105": {"class_type": "LoadLatent", "inputs": {"latent": audio_latent}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["104", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["105", 0], "vae": ["4", 0]}},
        "13": {"class_type": "VHS_VideoCombine",
               "inputs": {"images": ["11", 0], "audio": ["12", 0],
                          "frame_rate": FPS, "loop_count": 0,
                          "filename_prefix": out_prefix,
                          "format": VIDEO_FORMAT, "pix_fmt": "yuv420p",
                          "crf": 19, "save_metadata": True,
                          "trim_to_audio": False,
                          "pingpong": False, "save_output": True}},
    }


# --------------------------------------------------------------------------
# Reference roles and composed reference prompts
# --------------------------------------------------------------------------

VIDEO_ROLES = ("structure", "edit", "continue", "motion", "swap")
AUDIO_ROLES = ("music", "voice", "copy")


# **Do not put a specific attribute in a generic template.** The environment
# line said "architecture, palette, and lighting" until 2026-08-16, on every
# image-reference arm, for whatever image happened to be wired. Measured that
# day (docs/prompt_length_experiment.md): against a mountain-lake reference
# with no buildings in it, the arm whose detailed_description was silent about
# the environment rendered the man inside a timber veranda with a chalet beside
# it -- the word had nothing to contradict it, so it built one. The arm whose
# description named the actual lake and meadow produced no structure at all.
#
# The generator cannot see the reference, so it must only assert what is true
# of ANY environment. "setting" is; "architecture" is not. Naming the real
# content is the prompt author's job, and it is load-bearing rather than
# decorative -- a label is a bare ordinal and carries no meaning until
# something says what it is.
#: What each image-reference role asks of its picture, as (definition,
#: retention) with `{i}` for the subject/picture ordinal.
#:
#: **A role is declared by the caller, never inferred from the socket.**
#: `_ref_prompt` cannot see the file wired to a socket, so any relationship it
#: states is an assertion about content it has not looked at. This repo has
#: already paid for that: the environment template claimed "architecture" for
#: whatever image happened to be there, and a mountain-lake reference with no
#: buildings produced a timber veranda and a chalet (`1fa5607`,
#: `docs/prompt_length_experiment.md`). The graph author picked the file and is
#: the only one who knows what is in it, so the role travels with the graph.
#:
#: Markers follow guide 4.1. `attribute_transfer` is for a characteristic moved
#: onto a *different* subject, which is why the garment carries it and the
#: character does not.
_IMAGE_ROLE_PROSE = {
    "character": (
        "<Subject {i}> is the main character in <Picture {i}>, whose face, hair, and clothing are carried into the target video.",
        "<Subject {i}> (appears in [Shot 1]): fully_preserved - face, hair, and clothing are retained.",
    ),
    # Scoped to setting and deliberately NOT to occupants: an environment
    # plate may contain people, and a broader line puts them in competition
    # with <Subject 1>'s identity.
    "environment": (
        "<Subject {i}> is the environment in <Picture {i}>, which provides the setting for the target video.",
        "<Subject {i}> (appears in [Shot 1]): fully_preserved - the visual setting is retained.",
    ),
    "garment": (
        "<Subject {i}> is the garment shown in <Picture {i}>, which <Subject 1> wears in the target video.",
        "<Subject {i}> (appears in [Shot 1]): attribute_transfer - the garment from <Picture {i}> is placed on <Subject 1>.",
    ),
    # Last resort, and it asserts only what is true of any image reference.
    # Prefer adding a named role above over reaching for this.
    "subject": (
        "<Subject {i}> is an additional reference subject shown in <Picture {i}>, whose appearance is carried into the target video.",
        "<Subject {i}> (appears in [Shot 1]): fully_preserved - the appearance of <Subject {i}> is retained.",
    ),
}

#: What `images=True` has always meant. Named so the byte-identity of every
#: existing graph is a constant rather than a coincidence of ordering.
_DEFAULT_IMAGE_ROLES = ("character", "environment")

#: Every image role a graph may declare, exported for the same reason
#: `VIDEO_ROLES` is: `bench/check_ref_prompt_labels.py` reproduces every prompt
#: the generator can emit, and a hardcoded copy there stops covering this file
#: the moment a role is added.
#:
#: **Adding a role is not enough to make the check see it.** That check used to
#: enumerate `images` as `(True, False)`, which cannot express a role tuple at
#: all -- so the first graph to declare three roles read as a hand-edited
#: prompt. A new *form* of an argument is invisible to an enumeration written
#: for its old *values*; when the shape of an input changes, the enumerator
#: has to change with it.
IMAGE_ROLES = tuple(_IMAGE_ROLE_PROSE)


def _image_roles(images):
    """Normalise `images=` into a tuple of role names.

    Accepts the three spellings a caller can want, and nothing else:

      False / None          no image references
      True                  the historical pair, ("character", "environment")
      ("character", ...)    an explicit role per socket, in socket order

    An int is deliberately NOT accepted. `images=3` would have to invent roles
    for pictures it cannot see, which is the failure this table exists to
    prevent -- the caller wiring the files is the one who knows what they are.
    """
    if not images:
        return ()
    if images is True:
        return _DEFAULT_IMAGE_ROLES
    roles = tuple(images)
    unknown = [r for r in roles if r not in _IMAGE_ROLE_PROSE]
    if unknown:
        raise SystemExit(
            f"_ref_prompt: unknown image role(s) {unknown}. Known roles are "
            f"{sorted(_IMAGE_ROLE_PROSE)}. Add one to _IMAGE_ROLE_PROSE with "
            "prose written against the official guide rather than passing a "
            "role the table cannot render.")
    if not 1 <= len(roles) <= len(_REF_IMAGE_NODES):
        raise SystemExit(
            f"_ref_prompt: {len(roles)} image roles, but the builder wires "
            f"{len(_REF_IMAGE_NODES)} image sockets. These must match -- "
            "bench/check_ref_prompt_labels.py fails the build when the prompt "
            "names labels the graph does not wire.")
    return roles


def _role_label(image_roles, want):
    """`<Subject N>` for the first socket carrying `want`, or None."""
    for n, role in enumerate(image_roles, start=1):
        if role == want:
            return f"<Subject {n}>"
    return None


def _env_label(image_roles):
    """`<Subject N>` for the environment reference, or None if there isn't one.

    The shot prose has an establishing beat that puts <Subject 1> inside the
    scene, and it hard-coded `<Subject 2>` while that was the only arrangement
    the builder could express. With roles declared per socket the environment
    can sit anywhere, so this resolves it by role. Returns None when no socket
    carries `environment`, and the callers drop the beat rather than naming a
    subject that is not a place.
    """
    return _role_label(image_roles, "environment")


#: Ref-form bodies for the stress scenes: the `detailed_description` a ref2va
#: arm gets instead of the one-shot establishing beat the role tables build.
#:
#: **Written for the six-field layout, not transformed from the t2v text.** A
#: transformer would have to rewrite "a busker in her twenties" into "the
#: subject shown in <Picture 1>" and would produce a prompt asserting both.
#: These say the reference thing once, in the guide's own vocabulary.
#:
#: `{character}` and `{environment}` are filled with the labels the arm
#: actually wires. Every wired `<Subject N>` MUST appear in the result --
#: ref_en.md:231, and `bench/preflight_graph.py` warns when one is defined and
#: never cited -- so a scene that cannot cite a role is refused rather than
#: emitted with the label missing.
#:
#: Speaker ids stay OUT of retention_analysis; a `(Sx)` there is a hard fail
#: (ref_en.md:278). They belong here, in the description.
REF_SCENE_SHOTS = {
    "subway": [
        "[Shot 1] Handheld with fast reframing under cool platform fluorescents. "
        "A wide shot establishes {environment_beat}a crowded underground platform, "
        "tiled columns receding, a train braking into frame from the right, "
        "still moving fast. A tiled platform sign on the nearest column reads "
        "\"NORTHBOUND - PLATFORM 2\" in white capitals on a dark blue "
        "ground. <|caption_start|>NORTHBOUND - PLATFORM 2<|caption_end|> "
        "{character}, with a bright, slightly raw mezzo (S1), stands over an open "
        "guitar case, preserving the face, hair, "
        "wardrobe and build established in the reference, strums once, and sings "
        "into the arriving noise: <|lyrics_start|><d>[English] Nobody waits on the "
        "northbound line.</d><d>[English] Everybody's leaving on time.</d>"
        "<|lyrics_end|> Her lips close on the last word as the headlights wash "
        "across her face and commuters surge past in both directions, one man "
        "breaking into a run.",
        "[Shot 2] The shot cuts to a tight two-shot of two commuters "
        "shouldering fast through the crowd, the camera tracking with them at "
        "large amplitude. A woman in a soaked raincoat with a clipped, urgent "
        "contralto (S2) turns her head without slowing and says: <d>[English] Do "
        "not stop, it is the last one.</d> Her lips close. Her companion, a man in "
        "his thirties with a breathless, higher tenor (S3), answers half a step "
        "behind her: <d>[English] I know, I know, go, go.</d> His lips close and "
        "he shoves his bag under one arm as they cut left around a column, "
        "{character} out of frame behind them, still at her guitar case.",
        "[Shot 3] The camera whip pans to a low wide shot of the "
        "platform edge as the doors open and the crowd compresses inward, an "
        "umbrella knocked from a commuter's hand skidding across the tiles. "
        "{character} (S1) keeps playing "
        "through it, her identity unchanged from the reference, and sings over "
        "the crowd: <|lyrics_start|><d>[English] Hold the door and hold your line."
        "</d><|lyrics_end|> Her lips close. The camera holds wide long enough "
        "to keep the tiled columns, the platform edge markings and the "
        "overhead signage of the reference setting continuously visible behind "
        "her while the crowd moves across the frame in both directions, coats "
        "and bags passing close to the lens without occluding her face.",
        "[Shot 4] The shot changes to a close shot inside the "
        "carriage looking out through the closing doors, the woman in the raincoat "
        "(S2) pressed against the glass, breathing hard, calling back to her companion, "
        "who was cut off by the crowd and is still on the platform outside the "
        "closing doors, {character} small at her guitar case beyond him: "
        "<d>[English] Get the next one and meet me at the"
        "</d><|cutoff|>",
    ],
    "kitchen": [
        "[Shot 1] Handheld, fast reframing, hard practical light off stainless "
        "steel. A medium-wide shot establishes {environment_beat}a restaurant line "
        "mid-service, four burners lit, steam crossing the lens, a ticket rail "
        "loaded above the pass. The ticket closest to camera reads "
        "\"TABLE 12 - 2 COVERS - FIRE\" in narrow black type on white "
        "thermal paper. <|caption_start|>TABLE 12 - 2 COVERS - FIRE"
        "<|caption_end|> {character}, with a hard, carrying baritone (S1), works "
        "the pass, preserving the face, hair, "
        "wardrobe and build established in the reference, slaps the rail and calls "
        "down the line: <d>[English] Two on twelve, fire it now.</d> His lips "
        "close and he snaps the ticket free with two fingers. The camera tracks "
        "right at large amplitude and fast speed past three cooks, one tossing a "
        "pan so the flame climbs above the rim, the youngest working the flat top, "
        "and a runner waiting at the end of the pass.",
        "[Shot 2] The shot cuts to a close shot of the young line cook "
        "with a light, quick soprano (S2) at the flat top, moving fast, who "
        "answers without looking up: <d>[English] Two on twelve, heard.</d> Her "
        "lips close, and she sings along under her breath with a radio on the "
        "shelf behind her: <|lyrics_start|><d>[English] Keep it moving, keep it "
        "hot.</d><|lyrics_end|> Her lips close as she flips two portions in one "
        "motion and the flame flares behind her shoulder.",
        "[Shot 3] The camera pushes in fast with large amplitude on "
        "the pass as two plates land side by side, the hands of {character}, the "
        "young cook and the pan-tossing cook entering frame from three directions, "
        "the young cook's thumb wiping a rim clean. {character} (S1) and the young "
        "cook (S2), leaning in from the flat top, overlap "
        "with no gap between them: <d>[English] Where is my second plate.</d> <d>[English] Behind "
        "you, behind you.</d> Both sets of lips close as {character} spins a plate "
        "into position. The camera stays low across the pass so the stainless "
        "surfaces, the loaded ticket rail and the lit burners of the reference "
        "setting remain continuously visible behind the hands, steam crossing "
        "the lens twice without hiding either face.",
        # Shot 4 ran 50 words against the other three at 81-107, and was
        # missing two of the things ref 5.2 asks every shot to establish:
        # camera movement, and current sound. Filling those took the scene's
        # `detailed_description` from 349 words to inside the guide's 350-500
        # band -- the beat is here because the shot was underspecified, not to
        # clear the number. `Pedestal Up` is base 4.3's table entry for this
        # move; "cranes up" is not in it.
        "[Shot 4] The shot changes to a low shot as the runner lifts "
        "both plates and turns for the door, the kitchen receding behind him in a "
        "blur of steam. The camera pedestals up with small amplitude at slow "
        "speed as he passes, holding the lit burners and the loaded ticket rail "
        "of the reference setting across the top of the frame while the ticket "
        "printer starts another run behind the pass and the pan-tossing cook sets "
        "a pan down hard on the flat top. {character} (S1) calls after him already reading the next "
        "ticket: <d>[English] And tell them the special is</d><|cutoff|>",
    ],
}

#: Soundscape and score per stress scene in ref form -- the same sound world as
#: the t2v version, since the reference changes who is in the shot rather than
#: what the room sounds like.
REF_SCENE_AUDIO = {
    "subway": ("Brake squeal rising and cutting out as a train settles, a dense "
               "crowd shuffling and coats brushing, a single guitar strummed "
               "hard over the noise, an umbrella skittering across tile, a "
               "two-tone door chime, and pneumatic doors sealing with a hard "
               "thump.", "N/A"),
    "kitchen": ("A ticket printer chattering in bursts, a metal rail slapped "
                "flat, pans ringing on a flat top with sharp oil crackle, a gas "
                "burner whumping as it catches, plates set down hard in quick "
                "succession, and a thin radio behind everything.", "N/A"),
}


#: The identity reference each scene arm wires, one `character` picture per
#: scene. Chosen to match what the scene's own text already says about the
#: person, because a reference that contradicts the description is an arm
#: testing two things at once:
#:
#: * `subway` calls the busker "her" in Shot 3 ("keeps playing through it, her
#:   identity unchanged"), and `subject_performer_stage` is a woman performer.
#: * `kitchen` gives the line cook at the flat top a "light, quick soprano
#:   (S2)" and makes `{character}` a DIFFERENT person working the pass, so
#:   this picks a face that cannot be confused with S2.
#:
#: Deliberately `character` only, with no `environment` role. The one kitchen
#: environment asset on the box is papercraft, and these scenes are
#: photorealistic live-action -- wiring it would put a style transfer inside an
#: arm that exists to test markers and description length. Adding an
#: environment arm later is a second entry, not an edit to this one.
SCENE_REF_IMAGES = {
    "subway": ("h3_refs/subject_performer_stage_662x1177.png",),
    "kitchen": ("h3_refs/face_young_man_glasses_1024x1024.png",),
}



def _scene_description(scene: str, image_roles, defs) -> str:
    """A stress scene's `detailed_description`, with the labels this arm wires.

    `{character}` resolves by ROLE, not by ordinal, for the same reason the
    establishing beat does: once roles are declared per socket the character
    can sit anywhere, and an arm whose first socket is a garment would
    otherwise have the busker played by a coat.

    **Refuses rather than under-cites.** Every defined `<Subject N>` has to
    appear in `detailed_description` (ref_en.md:231); a subject that carries a
    retention marker and is never mentioned asks the model to transfer
    something onto nothing, and `bench/preflight_graph.py` warns about it after
    the fact. A scene that cannot cite every role this arm defines is a
    mismatch between the scene and the arm, so it stops the build instead.
    """
    import re as _re
    char = _role_label(image_roles, "character") or "<Subject 1>"
    env = _env_label(image_roles)
    env_beat = f"{env}, which supplies the setting for the target video, and "
    body = "\n".join(sh.format(character=char,
                               environment_beat=(env_beat if env else ""))
                     for sh in REF_SCENE_SHOTS[scene])
    defined = sorted(set(_re.findall(r"<Subject \d+>", " ".join(defs))))
    missing = [lab for lab in defined if lab not in body]
    if missing:
        raise SystemExit(
            f"_ref_prompt: scene {scene!r} never cites {missing}, which this "
            f"arm defines. Either the arm wires a role the scene has no part "
            f"for, or the scene needs a beat for it -- do not ship a defined "
            f"subject the description never mentions.")
    return body


#: **The dialogue probe, and the one arm here built to be JUDGED rather than to
#: look good.** Rendered 2026-08-08 as `marker_arm_vendortokens`
#: (`internal/refs/marker_arm_vendortokens_api.json`, clips under the owner's
#: `Video/20260808-stock-vs-vendortokens/`) and kept because it worked.
#:
#: **It is no longer a verbatim reproduction of what was rendered, and this
#: comment said it was until 2026-08-28.** `d5be353` replaced "the camera
#: drifts a few degrees" with "shakes slightly" here and in the ref2va twin,
#: because the original is not in base guide 4.3's closed camera table --
#: a guide correction, made without the paired render the last line of this
#: comment asks for. Diff this constant against the api json before treating
#: any of it as the rendered configuration.
#:
#: What makes it a good test, as opposed to a good clip:
#:
#: * **Eight short lines instead of two long ones**, across three shots, so the
#:   clip carries seven speaker changes in fifteen seconds. Lip sync has to be
#:   right repeatedly and at speed; a model that drifts is caught on the next
#:   line rather than at the end.
#: * **The pacing is written, not hoped for**, and as of 2026-08-28 it is
#:   written ONLY in the body: "answers immediately" x2 and "says at once" x2,
#:   now joined by "answers at once" on Shot 1's fourth line. Those five cover
#:   all five within-shot line transitions, which is why the change below was
#:   safe to make.
#:
#:   **A soundscape sentence carried this until 2026-08-28** -- "two speaking
#:   voices ... trading short clipped lines with almost no gap between them",
#:   identical here and in the ref2va twin -- and it was dropped from both.
#:   Two reasons, and the first is the operative one. Base guide 4.6 scopes
#:   `overall_soundscape` to ambient sound, physical action sounds and
#:   NON-VERBAL human sounds (breathing, laughter, panting); two speaking
#:   voices is verbal human sound, the category that enumeration excludes by
#:   naming its complement. Ref guide 6 reaches the same place independently.
#:   Second, it was redundant: the ref2va twin says the same thing a third
#:   time in its `summary`, and the fifth body cue closes the one transition
#:   the body had left uncovered.
#:
#:   **What stood here as fact and was not one.** "Remove those and the same
#:   lines come out spaced and unjudgeable" was a belief, not a result -- no
#:   arm has ever been rendered without them, so nothing in `bench/results/`
#:   speaks to it either way. It is recorded because it nearly blocked a
#:   guide-correct edit on the strength of sounding measured.
#: * **Every line is `<d>[English] ...</d>`**, which the prompt guide requires
#:   for all dialogue. The marker is the subject: if the tokenizer is not
#:   emitting 151669/151670 the model hears angle brackets and the word
#:   "English" as prose, and the tell is AUDIBLE -- it speaks them or slurs the
#:   line start. Being able to hear that is why this arm exists.
#: * **Two voices described by register** (measured female S1, lower gravelled
#:   male S2), so a swapped or blended speaker is obvious without a spectrogram.
#:
#: Do not "improve" it without rendering the result beside this one. That
#: instruction has now been crossed twice, both times for a guide correction
#: and both times disclosed here rather than quietly: `d5be353` for the camera
#: verb, and the 2026-08-28 soundscape drop above. **The arm has not been
#: re-rendered since either**, so the clips in
#: `Video/20260808-stock-vs-vendortokens/` are the old prompt's output and this
#: constant is not what produced them.
DIALOGUE_T2V_PROMPT = _bank_prompt("t2va_stairwell_dialogue")

#: The ref2va twin. **Same eight lines, same three shots, same cuts, same
#: pacing language** -- the only thing that changes is where the two people come
#: from, which is the axis this arm is for.
#:
#: The format changes with it and that is not cosmetic. t2v takes
#: `integrated_multimodal_description`; ref2va takes subject definitions,
#: retention markers and a shot list, and the external system prompts take
#: OPPOSITE corrections across that boundary -- so this is a rewrite into the
#: ref2va form, not the t2v string with pictures bolted on.
#:
#: The two references are deliberately far apart: an elderly man in a navy suit
#: against a brown studio backdrop, and a woman in a red dress in daylight. A
#: blended or swapped identity shows up in one frame instead of needing a crop.
#: Both `... is not present in the target video` lines are load-bearing -- each
#: still carries a background the stairwell must not inherit.
#:
#: **Subject definitions are BARE -- no `(Sx)` on the definition line.** The
#: speaker id belongs in the description, where the speech happens, and in an
#: `<Audio N>` definition when one maps to a subject. That is the guide's own
#: usage: its definitions read `<Subject 3> is the young blonde woman in
#: <Video 1>` while its description reads `<Subject 2> (S1) turns toward the
#: woman and says` (ref_en guide, lines 103 and 314-316). This prompt carried
#: `(S1)`/`(S2)` on the definition lines until 2026-08-26;
#: `bench/check_ref_prompt_labels.py` caught it, and the guide agreed with the
#: check rather than with the prompt.
DIALOGUE_REF2V_PROMPT = _bank_prompt("ref2va_stairwell_dialogue")


def _ref_prompt(*, images: bool | tuple[str, ...] = True,
                video=False, video_audio=False, audio=False,
                video_role="structure", audio_role="music", scene=None):
    """A ref2va prompt declaring EXACTLY the labels this arm wires, in the
    relationship it actually asks for.

    **The reference combination is mechanical; the relationship is the request.**
    Which records are appended decides which labels the tokenizer emits. What the
    prompt asks those labels to DO is a separate axis, and it is the one that
    changes the output. Every arm here used to be `structure` + `music`, the
    thinnest slice of what the guides describe.

    `video_role`, from official guide section 2.3, which names exactly three
    whole-video relationships plus the subject-sourcing rule in 2.1:

      edit       the source video for an edit. `partially_preserved`: keep the
                 framing, camera and timing, change what the prompt names.
                 **There is no mask socket on this node** -- the edit is
                 whole-frame regeneration conditioned on the source, so what
                 holds it together is `retention_analysis` saying precisely
                 what survives, not a painted region.
      continue   a continuation start point. The target begins where the
                 source ends.
      motion     motion transferred onto a DIFFERENT subject, via 2.1's
                 multi-asset subject ("appearance from <Picture 1>, walking
                 motion from <Video 1>") and the `attribute_transfer` marker.
                 Needs images, since something must receive the motion.
      structure  camera movement, cuts and rhythm only, at `weak_reference`.

    `audio_role`, from section 2.4:

      music      background-music style, at `reference`
      voice      a speaker's timbre and delivery, at `reference`, carrying the
                 `<Subject N> (Sx)` speaker id the guide requires
      copy       the track reused as the target's audio, at `fully_copy`

    Markers never cross sets: visual takes fully_preserved /
    partially_preserved / attribute_transfer / weak_reference (4.1), audio
    takes fully_copy / partially_copy / reference / weak_reference (4.2).

    **The result is returned from the bank, not from here** -- see
    `_composed_from_bank`. Every combination this function can reach that a
    graph actually asks for is a `prompt_bank/` entry, so a composed prompt
    is identifiable in a render record and gradeable by the same tool as
    every other, and a new combination fails the build until it is written
    down. What the composition decides is still what the text SAYS.
    """
    image_roles = _image_roles(images)
    defs, retention, shot = [], [], []
    audio_n = 0
    subject_from_video = video and not images

    if images and video and video_role == "swap":
        # Character replacement: the video is the PLATE and the image is the
        # new identity. Distinct from `edit` above, which keeps the person in
        # <Video 1> and changes what they wear -- here the person is what
        # changes and everything around them is what must not.
        #
        # The negative clauses are the whole technique and they are NOT in the
        # official guide, which never tells a reference what it does not
        # supply. They come from general prompting research, where the
        # reported failure is the model blending the two identities, or
        # dragging the image's lighting and background into the plate. Stated
        # as an untested hypothesis on purpose: this arm exists to find out
        # whether the negatives earn their tokens, and h3_ref_video_image_edit
        # is the twin to read it against.
        defs.append(
            "<Subject 1> is the character whose complete visual identity -- face, facial structure, eyes, skin tone, hair style and colour, body proportions, and overall appearance -- comes exclusively from <Picture 1>. Their body motion, posture, gestures, head movements, timing, and physical performance come from the original character in <Video 1>.")
        defs.append(
            "<Picture 1> supplies subject identity only. It does not supply lighting, exposure, colour grade, background, camera angle, pose, framing, or scene composition.")
        retention.append(
            "<Subject 1> (appears in [Shot 1]): fully_preserved - facial structure, identity, hair, and appearance from <Picture 1> are retained.")
    elif images and video and video_role == "edit":
        # The combination worth starting from for an edit: the VIDEO is the
        # source being altered and the IMAGE is what gets put into it. Without
        # the image the prompt has to describe the insert in words, which is
        # exactly the part a reference image is better at than prose.
        defs.append(
            "<Subject 1> is the person in <Video 1>, whose face, build, and position in frame are kept in the target video.")
        defs.append(
            "<Subject 2> is the garment shown in <Picture 1>, which replaces the one <Subject 1> wears in <Video 1>.")
        defs.append(
            "<Subject 3> is the environment in <Picture 2>, which replaces the background of <Video 1> while the camera move is kept.")
        retention.append(
            "<Subject 1> (appears in [Shot 1]): partially_preserved - face, build, posture, and motion are retained from <Video 1>; the garment and the background change.")
        retention.append(
            "<Subject 2> (appears in [Shot 1]): attribute_transfer - the garment from <Picture 1> replaces the original on <Subject 1>.")
        retention.append(
            "<Subject 3> (appears in [Shot 1]): fully_preserved - the visual setting comes from <Picture 2>.")
    elif images:
        if video and video_role == "motion":
            # 2.1: one subject, two assets, each named for what it provides.
            defs.append(
                "<Subject 1> is the person whose appearance comes from <Picture 1> and whose walking motion comes from <Video 1>.")
            defs.append(
                "<Subject 2> is the environment in <Picture 2>, which provides the setting for the target video.")
            # 4.1: attribute_transfer means "referenced characteristics are
            # transferred to a DIFFERENT identifiable target subject", so it
            # belongs on the source giving the trait away -- <Video 1> below.
            # On the recipient it reads as asking for this subject's own
            # appearance to move onto somebody else, the opposite request.
            retention.append(
                "<Subject 1> (appears in [Shot 1]): fully_preserved - face, hair, and clothing are retained from <Picture 1>.")
            retention.append(
                "<Subject 2> (appears in [Shot 1]): fully_preserved - the visual setting is retained.")
        else:
            # One line per wired socket, in socket order, from the role the
            # graph declared. `images=True` resolves to
            # ("character", "environment"), whose prose is byte-identical to
            # what this branch hard-coded before 2026-08-16 -- so every
            # existing graph regenerates unchanged, which is checked rather
            # than asserted (see the snapshot control in the commit).
            for i, role in enumerate(image_roles, start=1):
                line, ret = _IMAGE_ROLE_PROSE[role]
                defs.append(line.format(i=i))
                retention.append(ret.format(i=i))
    elif subject_from_video:
        if video_role == "edit":
            defs.append(
                "<Subject 1> is the person in <Video 1>, whose face, build, and position in frame are kept in the target video.")
            defs.append(
                "<Subject 2> is a bright red waxed-cotton jacket that replaces the garment <Subject 1> wears in <Video 1>.")
            retention.append(
                "<Subject 1> (appears in [Shot 1]): partially_preserved - face, build, posture, and motion are retained from <Video 1>; the garment changes.")
            retention.append(
                "<Subject 2> (appears in [Shot 1]): attribute_transfer - the red jacket replaces the original garment on <Subject 1>.")
        else:
            defs.append(
                "<Subject 1> is the person in <Video 1>, whose face, hair, and clothing are carried into the target video.")
            retention.append(
                "<Subject 1> (appears in [Shot 1]): fully_preserved - face, hair, and clothing are retained from <Video 1>.")

    if video_audio:
        audio_n += 1
        if audio_role == "copy":
            defs.append(f"<Audio {audio_n}> is the synchronized audio track of <Video 1> and is reused in the target video.")
            retention.append(f"<Audio {audio_n}>: fully_copy - <Audio {audio_n}> is reused 1:1 as the target video's complete final audio track.")
        else:
            defs.append(f"<Audio {audio_n}> is the synchronized audio track of <Video 1> and is reused in the target video.")
            retention.append(f"<Audio {audio_n}>: partially_copy - the ambience of <Audio {audio_n}> is kept under the new scene.")

    if video:
        role_def = {
            "swap": "<Video 1> is the source video for the target video edit. It supplies the camera path, framing, background, environment, lighting, composition, action timing, and the original character's body motion. It does not supply the face or identity.",
            "edit": "<Video 1> is the source video for the target video edit.",
            "continue": "<Video 1> is the source video the target video continues from, beginning at its final frame.",
            "motion": "<Video 1> is the source of the walking motion transferred to <Subject 1>; its own scene is not reused.",
            "structure": "<Video 1> is the source video whose camera movement the target video follows.",  # NOT "cutting rhythm": see role_ret below
        }[video_role]
        role_ret = {
            "swap": "<Video 1> (environment and motion): partially_preserved - the setting, lighting, and camera composition are retained, and the original character's actions are transferred to <Subject 1>.",
            "edit": "<Video 1> (source video for the edit): partially_preserved - framing, camera movement, and shot timing are kept; only what is named above changes.",
            "continue": "<Video 1> (continuation source): partially_preserved - scene, lighting, and subject position continue from its final state.",
            "motion": "<Video 1> (motion source): attribute_transfer - only the gait and its timing are taken; the scene and the person are not.",
            # Was "(cut and pacing structure) ... only the pacing" until
            # 2026-08-16, which asked for something the prompt did not
            # contain: every ref arm is a SINGLE shot whose summary says
            # "a single continuous shot", so a cut-structure reference had
            # no cuts to follow. Five graphs shipped that contradiction.
            # Narrowed to what the arm actually asks for. If these arms ever
            # gain a shot timeline, the cut language can come back with it.
            "structure": "<Video 1> (camera movement): weak_reference - only the path and pacing of the camera move is followed; its scene and cutting are not.",
        }[video_role]
        defs.append(role_def)
        retention.append(role_ret)

    if audio:
        audio_n += 1
        if audio_role == "voice":
            # 2.4 requires the target speaker's global id, not a new number.
            who = "<Subject 1>" if (images or subject_from_video) else "the speaker"
            defs.append(f"<Audio {audio_n}> is the voice-timbre reference for {who} (S1).")
            retention.append(f"<Audio {audio_n}>: reference - only timbre and delivery are referenced, the signal is not copied.")
        elif audio_role == "copy":
            defs.append(f"<Audio {audio_n}> is the audio asset reused as the target video's audio track.")
            retention.append(f"<Audio {audio_n}>: fully_copy - reused 1:1 as the target video's complete final audio track.")
        else:
            defs.append(f"<Audio {audio_n}> is a standalone music reference whose tempo and instrumentation the target video's score follows.")
            retention.append(f"<Audio {audio_n}>: reference - only tempo and instrumentation are referenced, the signal is not copied.")

    # The shot text has to cite each label where its relationship is active
    # (guide 5.3), not merely mention it once in the definitions.
    if video and video_role == "swap":
        # Environment first, then the swap. The ordering is the point: naming
        # the plate before the replacement is what the technique claims keeps
        # the image's own scene from leaking into it.
        shot.append("The scene maintains the exact environmental details, lighting, and composition of <Video 1>.")
        shot.append("Within this space, <Subject 1> performs the exact movements and actions of the original character from <Video 1>, executing every gesture, step, and head turn frame for frame, while the face, hair, and build stay those defined by <Picture 1>.")
    elif video and video_role == "edit":
        shot.append("The shot reproduces <Video 1> frame for frame in framing, camera movement, and timing.")
        if images:
            shot.append("<Subject 1> keeps their face, build, posture, and every step of their motion, but now wears <Subject 2> and moves through <Subject 3> instead of the original background.")
        elif subject_from_video:
            shot.append("<Subject 1> keeps their face, build, posture, and every step of their motion, but now wears <Subject 2>, whose waxed cotton catches the light differently as they turn.")
        else:
            shot.append("<Subject 1> keeps their position and motion while the wardrobe named above changes.")
    elif video and video_role == "continue":
        shot.append("The shot begins exactly where <Video 1> ends, on the same framing and lighting, and carries the motion forward without a cut.")
        shot.append("<Subject 1> continues walking out of frame to the right as the camera holds.")
    elif video and video_role == "motion":
        shot.append("A medium shot establishes <Subject 2>, then <Subject 1> enters from the left, walking with the gait and timing taken from <Video 1>.")
        shot.append("The camera trucks right with small amplitude at slow speed.")
    else:
        if images:
            # The establishing beat needs the ENVIRONMENT subject, which is not
            # always <Subject 2> once roles are declared per socket. Resolving
            # it by role rather than by ordinal is what stops a three-reference
            # arm reading "a medium shot establishes <the garment>".
            # Every DEFINED subject has to be cited in detailed_description --
            # `check_ref_prompt_labels` enforces it, and it is right to: a
            # subject that carries a retention marker and never appears in the
            # shot is asking the model to transfer something onto nothing.
            # The first version of the role work defined a garment, gave it
            # `attribute_transfer`, and never put it in the shot; the check
            # caught it the first time a graph exercised the path.
            env = _env_label(image_roles)
            worn = _role_label(image_roles, "garment")
            enters = (f"<Subject 1> enters from the left{f' wearing {worn}' if worn else ''}"
                      " and stops at the center of the frame.")
            shot.append(f"A medium shot establishes {env}, then {enters}" if env
                        else f"A medium shot frames the scene, then {enters}")
            # Anything with no scripted beat of its own still has to appear.
            for n, role in enumerate(image_roles, start=1):
                if role in ("character", "environment", "garment"):
                    continue
                shot.append(f"<Subject {n}> is visible in the shot.")
        elif subject_from_video:
            shot.append("A medium shot frames <Subject 1>, who enters from the left and stops at the center of the frame.")
        else:
            shot.append("A medium shot establishes a quiet interior, and a figure enters from the left and stops at the center of the frame.")
        shot.append("The camera trucks right with small amplitude at slow speed"
                    + (", holding the unhurried pace of <Video 1>." if video else "."))

    summary = {
        "swap": "The target video is an edited version of <Video 1>, replacing its original character with <Subject 1> from <Picture 1> while preserving the camera movement, environment, and audio",
        "edit": "The target video is an edited version of <Video 1>, keeping its framing and motion while replacing what the retention analysis names",
        "continue": "The target video continues <Video 1> from its final frame, without a cut",
        "motion": "The target video places <Subject 1> inside <Subject 2>, carrying the walking motion of <Video 1>",
        "structure": ((f"The target video places <Subject 1> inside {_env_label(image_roles)} for a single continuous shot"
                       if _env_label(image_roles) else
                       "The target video places <Subject 1> in a single continuous shot")
                      if images else "The target video places <Subject 1> in a single continuous shot"),
    }[video_role if video else "structure"]
    if not video and not images:
        summary = "The target video is a single continuous shot"
    if audio:
        summary += (f", with the voice of <Audio {audio_n}>" if audio_role == "voice"
                    else f", scored after <Audio {audio_n}>")

    # Guide 6: "Write complete dialogue and lyrics only inside `<d>` in
    # `detailed_description`; do not repeat them in these two sections."
    # The line therefore lives in the shot, and `overall_soundscape` states
    # only the relationship for its audible layer (guide 6, same paragraph).
    if audio and audio_role == "voice":
        who = "<Subject 1>" if (images or subject_from_video) else "A figure"
        shot.append(
            f"{who} (S1) turns toward the camera and says, in the clear timbre "
            f"referenced from <Audio {audio_n}>, "
            "<d>[English] I thought you would have gone by now.</d>")

    soundscape = "Natural ambient atmosphere continues throughout the shot."
    if video_audio:
        soundscape = "The ambience of <Audio 1> continues under the shot."
    if audio and audio_role == "voice":
        soundscape += (f" The vocal timbre of <Audio {audio_n}> is referenced for the "
                       "speaking voice, and its signal is not copied.")

    music = "N/A"
    if audio and audio_role == "music":
        music = f"A slow instrumental score follows the tempo and instrumentation of <Audio {audio_n}>."

    # ref guide section 3's task-type vocabulary (ref_en has no 3.2; base_en's 3.2 is FL2VA). This is NOT cosmetic: it is the only
    # place the prompt states what relationship the references stand in, and
    # every arm shipped `[reference generation]` regardless of role, which
    # collapsed the exact axis these arms exist to vary.
    #
    # ref section 3 is explicit that presence does not imply a type -- "if a reference
    # video provides only camera movement, cuts, or rhythm, it normally
    # belongs to `reference generation`" -- so motion and structure stay
    # reference generation and only edit/continue get their own type.
    types = []
    if video and video_role in ("edit", "swap"):
        # A character swap IS a direct modification of the source video, so
        # ref section 3 puts it here and not under `reference generation`. Community
        # write-ups of this scenario often stop at a bare `[video editing]`;
        # 3.2 is explicit that reused audible audio adds `audio reuse` too,
        # which the block below supplies.
        types.append("video editing")
    elif video and video_role == "continue":
        types.append("video continuation")
    if images or (video and video_role in ("motion", "structure")):
        types.append("reference generation")
    # 4.2's markers decide the audio type: fully_copy/partially_copy are a
    # reuse of the signal, `reference` is not. Ref section 3: "when editing a source
    # video, use `audio reuse` as well if its original audio remains audible."
    if video_audio or (audio and audio_role == "copy"):
        types.append("audio reuse")
    if audio and audio_role in ("voice", "music"):
        types.append("audio reference")
    if not types:
        types.append("reference generation")

    return _composed_from_bank("\n".join([
        "subject_definitions:", *defs, "",
        "summary:", f"[{' + '.join(types)}] " + summary + ".", "",
        "retention_analysis:", *retention, "",
        "detailed_description:",
        "The target video is in a cinematic live-action style.",
        # A scene replaces the one-shot establishing beat the role tables
        # build, and brings its own sound world with it -- a reference changes
        # WHO is in the shot, not what the room sounds like. The role
        # machinery above still owns subject_definitions, the summary and the
        # retention analysis, which is the half that has to match the sockets.
        (_scene_description(scene, image_roles, defs) if scene
         else "[Shot 1] " + " ".join(shot)), "",
        "overall_soundscape:",
        REF_SCENE_AUDIO[scene][0] if scene else soundscape, "",
        "non_diegetic_music:",
        REF_SCENE_AUDIO[scene][1] if scene else music,
    ]))


# --------------------------------------------------------------------------
_PKG_NAME = Path(__file__).resolve().parent.parent.name

# Static validation against /object_info
# --------------------------------------------------------------------------

def object_info_is_stale(oi: dict, source: str):
    """(verdict, details) for whether the served schema matches the code on disk.

    **Why the generator has to ask this itself.** Its report reads "validated N
    graphs against object_info: ok", and on 2026-08-22 that line printed while
    the running ComfyUI predated a schema change made minutes earlier -- so the
    graphs were validated against a schema that did not include a new input,
    and 34 of them turned out to disagree the moment the server was restarted.
    A green line that cannot distinguish "agrees with the current code" from
    "agrees with a stale server" teaches you to trust the wrong thing.

    Compares the input NAMES `/object_info` serves for each of this pack's
    nodes against the names their `define_schema` declares right now. Names
    only: types and defaults are `bench/check_schema_defaults.py`'s job, and
    this is a staleness signal rather than a second schema validator.

    Four verdicts, and the last two exist because the first version of this
    function had only two. It returned "matches" whenever the pack failed to
    import -- so a syntax error in a node file, which is one of the ways the
    server ends up stale in the first place, produced a clean green line.
    Found 2026-08-22 by a deliberate violation that broke a node's schema:
    the run reported ok because the import raised and the failure was
    swallowed. **Green because it could not look is the failure this repo
    names most often, and it went straight back in.**

      "matches"  compared, the server serves what the code declares
      "stale"    compared, they disagree; details say how
      "skipped"  a cached --object-info file, with no live server behind it,
                 so staleness is not a question that has an answer
      "blind"    the pack could not be imported or exposes no nodes, so
                 NOTHING was compared. Never report this as ok.
    """
    if not source.startswith("http"):
        return "skipped", ["--object-info is a cached file, so there is no "
                           "live server whose staleness could be checked"]
    try:
        import importlib
        # custom_nodes/ on the path so the pack resolves by its directory name,
        # which carries hyphens and is therefore not an importable identifier.
        _parent = str(Path(__file__).resolve().parents[2])
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
        pkg = importlib.import_module(_PKG_NAME)
    except Exception as exc:
        return "blind", [f"could not import {_PKG_NAME}: "
                         f"{type(exc).__name__}: {exc}"]
    ext = getattr(pkg, "comfy_entrypoint", None) or getattr(pkg, "NODES_LIST", None)
    if ext is None:
        return "blind", [f"{_PKG_NAME} exposes neither comfy_entrypoint nor "
                         f"NODES_LIST, so its nodes cannot be enumerated"]
    try:
        import asyncio, inspect
        val = ext() if callable(ext) else ext
        if inspect.iscoroutine(val):
            val = asyncio.run(val)
        nodes = val.get_node_list() if hasattr(val, "get_node_list") else val
        if inspect.iscoroutine(nodes):
            nodes = asyncio.run(nodes)
    except Exception as exc:
        return "blind", [f"could not enumerate this pack's nodes: "
                         f"{type(exc).__name__}: {exc}"]
    if not nodes:
        return "blind", ["this pack registered no nodes, so nothing was "
                         "compared against the server"]

    out = []
    compared = 0
    for node in nodes:
        try:
            schema = node.define_schema()
            nid = schema.node_id
            declared = {i.id for i in schema.inputs}
        except Exception as exc:
            out.append(f"{getattr(node, '__name__', node)}: define_schema "
                       f"raised, so it could not be compared: "
                       f"{type(exc).__name__}: {exc}")
            continue
        compared += 1
        served_entry = oi.get(nid)
        if served_entry is None:
            out.append(f"{nid}: this pack registers it, the server does not "
                       f"serve it -- the server has not loaded this code")
            continue
        served = set()
        for section in ("required", "optional"):
            served |= set((served_entry.get("input") or {}).get(section, {}))
        if declared - served:
            out.append(f"{nid}: code declares {sorted(declared - served)}, "
                       f"the server does not serve them")
        if served - declared:
            out.append(f"{nid}: the server serves {sorted(served - declared)}, "
                       f"the code no longer declares them")
    if compared == 0:
        return "blind", ["no node's schema could be read, so nothing was "
                         "compared"] + out
    if out:
        return "stale", out
    return "matches", [f"{compared} node schema(s) match the served schema"]


def load_object_info(source: str) -> dict:
    if source.startswith("http"):
        with urllib.request.urlopen(source.rstrip("/") + "/object_info", timeout=60) as r:
            return json.loads(r.read())
    return json.loads(Path(source).read_text())


# Inputs whose node declares VALIDATE_INPUTS and checks the filesystem instead
# of the combo list; add one when its node is read, not on the assumption that
# other loaders behave the same way. `LoadLatent.latent` read 2026-09-26: its
# `VALIDATE_INPUTS` and `load` both go through `folder_paths`' annotated-path
# functions (`ComfyUI/nodes.py::LoadLatent`), so `latents/x.latent [output]`
# resolves in the output directory.
_ANNOTATED_INPUTS = {("LoadImage", "image"), ("LoadLatent", "latent")}


def _annotated_path(class_type: str, name: str, val) -> bool:
    """True when this input legally takes a path the combo list does not offer.

    **This validator was stricter than the server, which is the same defect as
    being looser -- the direction differs, not the class.** `LoadImage`
    populates its combo from a NON-RECURSIVE `os.listdir` of the input
    directory (`ComfyUI/nodes.py`, `LoadImage.INPUT_TYPES`), so a file in a subfolder
    never appears in `/object_info`. But the node also defines
    `VALIDATE_INPUTS` -> `folder_paths.exists_annotated_filepath`, and
    ComfyUI's executor SKIPS its own combo check for any input the node
    validates itself. So `h3_refs/face_x.png` executes cleanly and this file
    was rejecting it.

    Verified by reading both, 2026-08-16, not inferred from behaviour:
    `ComfyUI/nodes.py::LoadImage.VALIDATE_INPUTS` and
    `ComfyUI/folder_paths.py::exists_annotated_filepath`, which joins the name under
    the input dir, refuses traversal, and returns `os.path.exists`.

    Membership is still checked for every bare filename -- the escape hatch is
    only for values carrying a subfolder, which is exactly the case
    `/object_info` cannot see. A typo in a root-level filename still fails.

    What this does NOT do is confirm the file exists; that needs the server's
    input directory, which this generator does not have. `bench/smoke_h3.py`
    submits and would surface a missing reference as a server-side rejection.
    """
    return (class_type, name) in _ANNOTATED_INPUTS and isinstance(val, str) \
        and "/" in val


def _combo_options(spec):
    """Combo option lists come in two shapes across ComfyUI node versions."""
    t = spec[0]
    if isinstance(t, list):
        return t
    if t == "COMBO":
        return (spec[1] or {}).get("options")
    return None


def core_sol_defaults_drift(oi: dict) -> list[str]:
    """Where `h3_config.SOL_CORE_DEFAULTS` disagrees with core's served defaults.

    The core-node probe renders ComfyUI's `BlockSparseAttention` AS CORE SHIPS
    IT, so its inputs must be the node's own schema defaults. The constant is a
    copy of those, and this is what stops the copy going quiet when a core
    release moves one: every validating build reads the defaults back out of
    /object_info and compares. The DynamicCombo branch's own inputs
    (`selection.tau`) are read from the option the constant selects.
    """
    node = oi.get(SOL_CORE_NODE)
    if node is None:
        return [f"{SOL_CORE_NODE} is not served, so the core-node probe cannot run here"]
    spec = node.get("input") or {}
    inputs = dict(spec.get("required") or {}) | dict(spec.get("optional") or {})
    served: dict = {}
    for name, s in inputs.items():
        meta = s[1] if len(s) > 1 and isinstance(s[1], dict) else {}
        if s[0] == "COMFY_DYNAMICCOMBO_V3":
            want = SOL_CORE_DEFAULTS.get(name)
            opt = next((o for o in meta.get("options", []) if o.get("key") == want), None)
            if opt is None:
                continue
            served[name] = want
            branch = opt.get("inputs") or {}
            for sub, ss in (dict(branch.get("required") or {})
                            | dict(branch.get("optional") or {})).items():
                sub_meta = ss[1] if len(ss) > 1 and isinstance(ss[1], dict) else {}
                if "default" in sub_meta:
                    served[f"{name}.{sub}"] = sub_meta["default"]
        elif "default" in meta:
            served[name] = meta["default"]
    missing = object()
    return [f"{SOL_CORE_NODE}.{k}: h3_config.SOL_CORE_DEFAULTS has {v!r}, the server "
            f"serves {served.get(k, '<no default>')!r}"
            for k, v in SOL_CORE_DEFAULTS.items() if served.get(k, missing) != v]


def validate_api(graph: dict, oi: dict, label: str) -> list[str]:
    errs = []

    def e(msg):
        errs.append(f"{label}: {msg}")

    for nid, node in graph.items():
        ct = node["class_type"]
        if ct not in oi:
            e(f"node {nid}: unknown class_type {ct!r}")
            continue
        spec = oi[ct]["input"]
        req = spec.get("required") or {}
        opt = spec.get("optional") or {}
        known = dict(req) | dict(opt)
        # Autogrow inputs are declared once but addressed as
        # "<input>.<prefix><i>"; expand the legal names.
        for name, s in list(known.items()):
            if s[0] != "COMFY_AUTOGROW_V3":
                continue
            tpl = (s[1] or {}).get("template") or {}
            inner = tpl.get("input") or {}
            inner_spec = next(iter((inner.get("required") or inner.get("optional") or {}).values()), None)
            for i in range(tpl.get("max", 0)):
                known[f"{name}.{tpl['prefix']}{i}"] = inner_spec

        # Format-dependent widgets. VHS_VideoCombine declares `format` as a
        # combo whose spec carries a per-format widget list (pix_fmt, crf,
        # save_metadata, ...), and reads those from **kwargs at run time --
        # `apply_format_widgets` warns and substitutes a default for any it
        # does not find. They are real inputs that /object_info does not
        # declare as inputs, so a plain known-name check calls every one of
        # them unknown. Third false-positive class this validator has had, all
        # the same shape: a node whose input set is not fully static.
        for parent, spec in list(req.items()) + list(opt.items()):
            meta = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            for widgets in (meta.get("formats") or {}).values():
                for w in widgets:
                    if isinstance(w, list) and w and isinstance(w[0], str):
                        known.setdefault(w[0], None)
            # A DynamicCombo declares each option's inputs nested under
            # `options` rather than as top-level inputs, and the API prompt
            # addresses them by their DOTTED path: `shape.wide_resolution`.
            #
            # This registered the BARE name until 2026-08-13, on the belief
            # that "the API prompt carries them flat for ComfyUI to re-nest".
            # It does not. ComfyUI's executor rejects the flat spelling with
            # `required_input_missing` naming `shape.wide_resolution`, so this
            # validator was passing graphs the server refuses -- every API
            # graph in the repo, for as long as the Resolution node has been
            # wired into them. A validator that accepts what the server
            # rejects is worse than no validator: it is a green light for a
            # graph that cannot run. Caught by `bench/smoke_h3.py` against a
            # live server, which is the only thing here that actually submits.
            #
            # A member of the option the graph selects is graded against its
            # own spec, so `extent.seconds` and `size_policy.*` get the type,
            # option and range checks a top-level input gets. Until 2026-09-14
            # every member was registered with no spec and graded on nothing.
            # A member of an option not selected stays known and ungraded.
            chosen = node["inputs"].get(parent)
            for option in (meta.get("options") or []):
                inner = (option.get("inputs") or {}) if isinstance(option, dict) else {}
                selected = isinstance(option, dict) and option.get("key") == chosen
                for section in ("required", "optional"):
                    for name, inner_spec in (inner.get(section) or {}).items():
                        if selected:
                            known[f"{parent}.{name}"] = inner_spec
                        else:
                            known.setdefault(f"{parent}.{name}", None)

        given = node["inputs"]
        # A DynamicCombo member must come after its parent in `inputs`: the
        # editor rebuilds every member when it sets the parent, so a member set
        # first is dropped when the file is loaded. Read from the frontend's
        # `core/graph/widgets/dynamicWidgets.ts` (comfyui_frontend_package
        # 1.52.7), not measured. Autogrow sockets are a different mechanism.
        earlier = set()
        for name in given:
            parent = name.split(".", 1)[0]
            if ("." in name and parent not in earlier
                    and (known.get(parent) or [None])[0] != "COMFY_AUTOGROW_V3"):
                e(f"node {nid} ({ct}): {name!r} comes before its parent {parent!r}, "
                  f"which the editor drops on load")
            earlier.add(name)
        for name in req:
            if req[name][0] == "COMFY_AUTOGROW_V3":
                continue
            if name not in given:
                e(f"node {nid} ({ct}): missing required input {name!r}")
        for name, val in given.items():
            if name not in known:
                e(f"node {nid} ({ct}): unknown input {name!r}")
                continue
            s = known[name]
            if isinstance(val, list):  # a link
                src, slot = val
                if src not in graph:
                    e(f"node {nid} ({ct}).{name}: links to missing node {src!r}")
                    continue
                souts = oi[graph[src]["class_type"]]["output"]
                if slot >= len(souts):
                    e(f"node {nid} ({ct}).{name}: output slot {slot} out of range "
                      f"on node {src} ({graph[src]['class_type']})")
                    continue
                got = souts[slot]
                want = s[0] if s else None
                got_name = got if isinstance(got, str) else "COMBO"
                if want and isinstance(want, str) and want not in ("*",) and got_name != want:
                    e(f"node {nid} ({ct}).{name}: type {got_name} from node {src} "
                      f"does not match {want}")
                continue
            if s is None:
                continue
            # The value's type, as the input's widget holds it. The executor
            # casts a `True` into an INT without a word, so only this refuses
            # it. As lenient as the UI validator it replaces: a FLOAT takes an int.
            want = s[0] if isinstance(s[0], str) else "COMBO"
            fits = {"BOOLEAN": isinstance(val, bool),
                    "INT": isinstance(val, int) and not isinstance(val, bool),
                    "FLOAT": isinstance(val, (int, float)) and not isinstance(val, bool),
                    "STRING": isinstance(val, str),
                    "COMBO": isinstance(val, (str, int, float, bool))}.get(want, True)
            if not fits:
                e(f"node {nid} ({ct}).{name}: {val!r} is not a {want} value")
                continue
            opts = _combo_options(s)
            if opts is not None and val not in opts and not _annotated_path(ct, name, val):
                e(f"node {nid} ({ct}).{name}: {val!r} is not an available option")
                continue
            meta = s[1] if len(s) > 1 and isinstance(s[1], dict) else {}
            if s[0] in ("INT", "FLOAT") and isinstance(val, (int, float)):
                if "min" in meta and val < meta["min"]:
                    e(f"node {nid} ({ct}).{name}: {val} below min {meta['min']}")
                if "max" in meta and val > meta["max"]:
                    e(f"node {nid} ({ct}).{name}: {val} above max {meta['max']}")

        # H3-specific: frame count is snapped up to 17k+5 by the node, so an
        # off-grid `length` silently renders a different duration than asked.
        if ct in ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo",
                  "MiniMaxH3Conditioning", "MiniMaxH3ReferenceConditioning",
                  "EmptyMiniMaxH3LatentAV"):
            ln = given.get("length")
            if isinstance(ln, int) and ln % 17 != 5:
                e(f"node {nid} ({ct}): length {ln} is off the 17k+5 grid; "
                  f"the node will snap it up to {ln + (5 - ln % 17) % 17}")

    # The mistake this whole file exists to prevent.
    #
    # A two-stage split legitimately has TWO model paths -- that is the point
    # of it -- so the invariant becomes: at most one source per stage, and a
    # second source is only allowed when SplitSigmas is actually present. That
    # keeps the check able to fail: without the SplitSigmas condition, adding a
    # stray second model path to an ordinary graph would now pass.
    split_nodes = [nid for nid, n in graph.items()
                   if n["class_type"] == "SplitSigmas"]
    # An audio-only refine pass (audio_refine.py) is a second sampler with its
    # own model path by design. Its scheduler and guider must agree with EACH
    # OTHER, and are taken out of the main pass's pool; anything else reading a
    # third source still fails below.
    refine_ids = set(refine_scheduler_ids(graph))
    for n in graph.values():
        if (n["class_type"] == "SamplerCustomAdvanced"
                and isinstance(n["inputs"].get("sigmas"), list)
                and str(n["inputs"]["sigmas"][0]) in refine_ids):
            sid, gid = str(n["inputs"]["sigmas"][0]), str(n["inputs"]["guider"][0])
            pair = {tuple(graph[x]["inputs"]["model"]) for x in (sid, gid)}
            if len(pair) != 1:
                e(f"refine pass: scheduler {sid} and guider {gid} read MODEL from "
                  f"different sources {pair}")
            refine_ids.add(gid)
    # A step-switch second pass (build_api(step_switch=True), route 3 of
    # docs/research/2026-09-26_distill_routing.md) is also a second sampler on
    # its own model by design. Recognised by structure, not by name, so the
    # exemption stays narrow: no added noise (DisableNoise), its own
    # ManualSigmas, and its latent is another sampler's output.
    for n in graph.values():
        if n["class_type"] != "SamplerCustomAdvanced":
            continue
        noise, li, sig, gd = (n["inputs"].get(k) for k in ("noise", "latent_image", "sigmas", "guider"))
        if (all(isinstance(x, list) for x in (noise, li, sig, gd))
                and graph[str(noise[0])]["class_type"] == "DisableNoise"
                and graph[str(li[0])]["class_type"] == "SamplerCustomAdvanced"
                and graph[str(sig[0])]["class_type"] == "ManualSigmas"):
            refine_ids.add(str(gd[0]))
    consumers = [(nid, n) for nid, n in graph.items()
                 if n["class_type"] in ("BasicScheduler", "BasicGuider")
                 and nid not in refine_ids]
    srcs = {tuple(n["inputs"]["model"]) for _, n in consumers
            if isinstance(n["inputs"].get("model"), list)}
    if split_nodes and len(srcs) == 2:
        # Both halves must still read sigmas from the SAME BasicScheduler --
        # one schedule cut in two is the precondition the whole split rests on.
        sched_ids = {nid for nid, n in graph.items()
                     if n["class_type"] == "BasicScheduler"}
        if len(sched_ids) != 1:
            e(f"split graph has {len(sched_ids)} BasicScheduler nodes; both "
              "stages must read one schedule or they are integrating "
              "different curves")
        for nid in split_nodes:
            src = graph[nid]["inputs"].get("sigmas")
            if not (isinstance(src, list) and src[0] in sched_ids):
                e(f"node {nid} (SplitSigmas): sigmas do not come from the "
                  "graph's BasicScheduler")
        srcs = set()          # two sources are expected here; checked above
    if len(srcs) > 1:
        e(f"BasicScheduler and BasicGuider read MODEL from different sources {srcs}; "
          f"one of them is bypassing a model patch")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object-info", default="http://127.0.0.1:8188",
                    help="running ComfyUI base URL, or a path to a saved object_info.json")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--chain", choices=sorted(DENSE_CHAINS), default=DEFAULT_DENSE_CHAIN,
                    help="dense kernel under Sol for every graph that names none "
                         f"(default {DEFAULT_DENSE_CHAIN!r}, h3_config.DEFAULT_DENSE_CHAIN). "
                         "Any other chain writes ONLY those graphs, needs --out "
                         "pointing outside the shipped tree, and skips the bench copies.")
    # Loading the right prompt into the right arm, without opening a JSON.
    # The graphs already ship with theirs baked in; these are for pasting one
    # into a graph you are editing by hand, or reading one without ComfyUI.
    ap.add_argument("--dump-prompts", action="store_true",
                    help="JSON map of shipped api filename -> its prompt, for "
                         "checks that compare a graph against ITS OWN expected "
                         "text rather than against every legal prompt")
    ap.add_argument("--dump-graphs", action="store_true",
                    help="JSON map of shipped api path (relative to workflows/) "
                         "-> {label, task, note}, for the generated workflows "
                         "index; nothing is built")
    ap.add_argument("--list-prompts", action="store_true",
                    help="one line per shipped graph: its name and prompt's first line")
    ap.add_argument("--print-prompt", metavar="GRAPH",
                    help="print one graph's exact prompt to stdout, ready to paste "
                         "(name may omit the h3_ prefix and the .json suffix)")
    ap.add_argument("--print-scene", metavar="NAME",
                    help="print one baseline t2v scene as a JSON string, which is "
                         "what `bench/run_graph_arms.py --set` parses: "
                         "--set \'arm:MiniMaxH3Conditioning.prompt=\'\"$(... --print-scene clinic)\"")
    ap.add_argument("--list-scenes", action="store_true",
                    help="the baseline scene names, in sweep order")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    written = []

    # The ones you actually open in ComfyUI. Named for what they do, not for
    # the task abbreviation the code uses internally.
    # `label` names the graph in the build and has to be unique; `task` is what
    # the builder dispatches on. They are separate because a task can have more
    # than one graph, differing only in model source.
    #
    # `extra` is the whole difference between a graph and the shipped one for
    # its task. Keeping it to one dict, on one line, next to the graph it
    # modifies is the point: graphs meant to be compared have to show what
    # differs in one place. Everything not
    # in `extra` -- seed, prompt, canvas, length, sampler, sage, Sol -- is
    # shared by construction, with ONE exception since 2026-08-13: an i2v
    # graph under the new `match_keyframe` default derives its canvas from the
    # loaded keyframe at run time, so its width/height are inert and it is not
    # canvas-comparable to the t2v and r2v graphs. See `_check_geometry`.
    # shared by construction and cannot drift apart.
    GRAPHS: tuple[tuple[str, str, str, str | None, dict[str, Any], str], ...] = (
        ("h3_text_to_video.json", "t2v", "t2v", LONG_T2V_PROMPT, {},
         "text -> video + audio"),
        # The audio-freeze lane's first graph (docs/h3_audio_freeze.md,
        # section 4 step 1). One node between the preflight and the sampler
        # is the whole difference from the graph above, plus the muxer
        # taking the node's own slice of the track in place of the audio
        # decoder. The prompt is the shipped t2v scene for now, which does
        # NOT describe the placeholder track; prompting the audio as it is
        # belongs to step 2 and needs a bank entry.
        ("h3_text_to_video_audio_freeze.json", "t2v-audio-freeze", "t2v",
         LONG_T2V_PROMPT,
         dict(freeze_audio=True, out_prefix="Video/h3_t2v_audio_freeze"),
         "text + a frozen audio track -> video, the track muxed as given"),
        # First-frame twin (plan step 5): the same node on the i2v chain, so
        # a window can be anchored on a frame. Canvas from the keyframe as
        # the shipped i2v graph does.
        ("h3_first_frame_to_video_audio_freeze.json", "i2v-audio-freeze", "i2v", None,
         dict(freeze_audio=True, out_prefix="Video/h3_i2v_audio_freeze"),
         "first frame + text + a frozen audio track -> video"),
        # The whole-track node (owner, 2026-09-12 evening: "if there's a way
        # to make it not manual"). The reframed dancer prompt on every
        # window, a 30-second look by default.
        ("h3_text_to_video_audio_freeze_song.json", "t2v-audio-freeze-song", "t2v",
         _bank_prompt("t2va_studio_dancer_close"),
         dict(freeze_song=True, freeze_song_seconds=30.0,
              freeze_context=39, out_prefix="Video/h3_t2v_audio_freeze_song",
              length=LONG_LENGTH),
         "a whole track from one node: windows planned from the song, one prompt throughout"),
        # The whole-track graph on the PDD8 baked chain, at the settings the
        # evidence supports for a full song (owner's ask, 2026-09-13 evening):
        # 345-frame windows with a 39-frame context (the only context judged:
        # the seam read better than either single window), the whole track,
        # the loose mask (read slightly better than frozen on the dancer at
        # two seeds; music, no speech), PDD8 at its own eight evaluations
        # (five forced an envelope tiling nothing has judged). Ships one bank
        # prompt and no timeline; paste a timeline and one `--- label` block
        # per label to line the windows up with the song's sections.
        ("h3_text_to_video_audio_freeze_song_pdd8.json", "t2v-audio-freeze-song-pdd8", "t2v",
         _bank_prompt("t2va_studio_dancer_close"),
         dict(pdd=True, sampler_name="euler",
              unet=MODELS["unet_fl2va_pdd8_baked"],
              lora=(PDD_FL2VA_STRIPPED_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              freeze_song=True, freeze_song_seconds=None,
              freeze_mask=0.25, freeze_context=39, length=LONG_LENGTH,
              out_prefix="Video/h3_t2v_audio_freeze_song_pdd8"),
         "a whole song on PDD8: 345-frame windows, 39 context, loose mask"),
        # The PDD8 song graph with the subject anchored by a reference still
        # (owner, 2026-09-14): fl2va takes references, and a fixed still is
        # the anchor a long song wants, not the previous window's last frame,
        # which would carry drift forward. The dancer scene's still and its
        # reference prompt from `h3_config.REFVIEW2_SCENES`; everything else
        # is the graph above.
        ("h3_text_to_video_audio_freeze_song_ref_pdd8.json", "t2v-audio-freeze-song-ref-pdd8", "t2v",
         _bank_prompt(next(p for t, p, _s in REFVIEW2_SCENES if t == "dancer")),
         dict(pdd=True, sampler_name="euler",
              unet=MODELS["unet_fl2va_pdd8_baked"],
              lora=(PDD_FL2VA_STRIPPED_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              freeze_song=True, freeze_song_seconds=None,
              freeze_song_refs=next(s for t, _p, s in REFVIEW2_SCENES if t == "dancer"),
              freeze_mask=0.25, freeze_context=39, length=LONG_LENGTH,
              out_prefix="Video/h3_t2v_audio_freeze_song_ref_pdd8"),
         "a whole song on PDD8 with the subject anchored by a reference still"),
        # The PDD8 song graph with prompt lists, the shipped example of
        # `__name__` placeholders (owner, 2026-09-14), on the owner's track
        # `just-a-flicker.mp3`, with the song's sections as the timeline. One
        # prompt for every section; the lists fill its second shot and move on
        # once per section, so every window of a section shares a place and
        # the place changes where the song does. Each window opens on a
        # close-up with the background lost to blur, so a seam into a new
        # place lands on her face. The song's lead is a breathy female vocal
        # over a slow lo-fi beat, per an analysis the owner supplied; the
        # prompt leaves the lyrics out, the untold form the lane found works
        # (`docs/h3_audio_freeze.md` step 4).
        ("h3_text_to_video_audio_freeze_song_lists_pdd8.json", "t2v-audio-freeze-song-lists-pdd8", "t2v",
         _bank_prompt("t2va_song_flicker_lists"),
         dict(pdd=True, sampler_name="euler",
              unet=MODELS["unet_fl2va_pdd8_baked"],
              lora=(PDD_FL2VA_STRIPPED_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              freeze_song=True, freeze_song_seconds=None, freeze_song_timeline=_SONG_FLICKER_TIMELINE,
              freeze_song_lists=_SONG_FLICKER_LISTS, freeze_track="just-a-flicker.mp3",
              freeze_mask=0.25, freeze_context=39, length=LONG_LENGTH,
              out_prefix="Video/h3_t2v_audio_freeze_song_lists_pdd8"),
         "a whole song on PDD8 on the song's sections, two prompt lists moving on once per section"),
        # Masked video-to-video on the PDD8 song chain (owner, 2026-10-04):
        # the track is a source video's own audio, SAM 3 tracks one subject,
        # and each window keeps the source outside that mask and regenerates
        # the subject from a reference still. The hard freeze, which is the
        # lane's default for a voice (`docs/h3_audio_freeze.md` step 4). One
        # prompt for every window that names no shot, because the plate
        # already holds the framing and the cuts; a clip-specific prompt is
        # an arm (`bench/masked_v2v_arms.json`). Not judged yet.
        # Since 2026-10-06 the prompt is written by the Masked Prompt node
        # (`masked_prompt.py`) and the bank text is its copy: the generic
        # swap text, which claims no count of people, duration, posture or
        # camera, where `ref2va_masked_subject_swap` said one performer
        # standing through a fourteen-second static shot.
        ("h3_video_to_video_masked_song_pdd8.json", "v2v-masked-song-pdd8", "t2v",
         _bank_prompt("ref2va_masked_person_swap"),
         dict(pdd=True, sampler_name="euler",
              unet=MODELS["unet_fl2va_pdd8_baked"],
              lora=(PDD_FL2VA_STRIPPED_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              freeze_song=True, freeze_song_seconds=30.0,
              freeze_song_refs=(PLACEHOLDER_IMAGE_A,), freeze_song_source=True,
              masked_prompt=MASKED_PROMPT,
              freeze_mask=0.0, freeze_context=39, length=LONG_LENGTH,
              out_prefix="Video/h3_v2v_masked_song_pdd8"),
         "masked video to video: a source video's subject replaced from a reference still, audio kept"),
        # A part of the subject through Sapiens2 (2026-10-06): the part node
        # labels the tracked subject and its mask is the region, in place of
        # the SAM phrases `head and hair` asks for. At the part node's
        # defaults that is hair, face and neck, on the original's body and
        # clothes, so the body's movement is the source's own and the fast
        # chain serves. Ticking another part on the node changes the region;
        # the prompt node's `picture_gives` then has to say what the still
        # provides. Not rendered.
        ("h3_video_to_video_masked_parts_song_pdd8.json", "v2v-masked-parts-song-pdd8", "t2v",
         _bank_prompt("ref2va_masked_person_head"),
         dict(pdd=True, sampler_name="euler",
              unet=MODELS["unet_fl2va_pdd8_baked"],
              lora=(PDD_FL2VA_STRIPPED_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              freeze_song=True, freeze_song_seconds=30.0,
              freeze_song_refs=(PLACEHOLDER_IMAGE_A,), freeze_song_source=True,
              masked_source=MASKED_PARTS_SOURCE, masked_parts=True, masked_prompt=MASKED_PARTS_PROMPT,
              freeze_mask=0.0, freeze_context=39, length=LONG_LENGTH,
              out_prefix="Video/h3_v2v_masked_parts_song_pdd8"),
         "masked video to video, a part of the subject: head and hair found by Sapiens2 and replaced from a still"),
        # The look before a masked render (2026-10-06, the masking board's
        # dry run): the default masked graph with the song node on `preview`,
        # so nothing samples and no model loads, and with what the render
        # graphs leave unwired: the tracker's numbered tiles and shot table
        # saved, the Masked Source's region shown. The prompt node shows the
        # text. The mask is kept for the render graphs.
        ("h3_video_to_video_masked_review.json", "v2v-masked-review", "t2v",
         _bank_prompt("ref2va_masked_person_swap"),
         dict(pdd=True, sampler_name="euler",
              unet=MODELS["unet_fl2va_pdd8_baked"],
              lora=(PDD_FL2VA_STRIPPED_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              freeze_song=True, freeze_song_seconds=30.0,
              freeze_song_refs=(PLACEHOLDER_IMAGE_A,), freeze_song_source=True,
              masked_prompt=MASKED_PROMPT, masked_review=True,
              freeze_mask=0.0, freeze_context=39, length=LONG_LENGTH,
              out_prefix="Video/h3_v2v_masked_review"),
         "masked video to video, the look before a render: who was tracked in each shot, the region and the prompt, nothing sampled"),
        # The same lane on the ref2va base for a shot that needs the original's
        # movement (owner, 2026-10-05, on the masking board): the subject's own
        # frames on grey reach the encoder as <Video 1> and the prompt ties the
        # subject's motion to it. Measured: ref2va carries the turn at
        # MASKED_MOTION_STEPS and loses it at 8, bake or not; fl2va never takes
        # the reference (bench/results/2026-10-05_masked_v2v_motion_arms.md).
        # Costs the step count over the PDD8 graph; not the default.
        # The Masked Prompt node writes the <Video 1> lines because the
        # Masked Source's motion reference is on. The measured text said
        # "the man"; this graph holds a placeholder still, so it ships the
        # same text for "the person" (`ref2va_masked_subject_motion` keeps
        # the measured one, and `bench/check_masked_prompt.py` holds the node
        # to it).
        ("h3_video_to_video_masked_song_ref2va_motion.json", "v2v-masked-song-ref2va-motion", "t2v",
         _bank_prompt("ref2va_masked_person_motion"),
         dict(sampler_name="euler", unet=MODELS["unet_ref2va"], steps=MASKED_MOTION_STEPS,
              freeze_song=True, freeze_song_seconds=30.0,
              freeze_song_refs=(PLACEHOLDER_IMAGE_A,), freeze_song_source=True,
              masked_source=MASKED_MOTION_SOURCE, masked_prompt=MASKED_PROMPT,
              freeze_mask=0.0, freeze_context=39, length=LONG_LENGTH,
              out_prefix="Video/h3_v2v_masked_song_ref2va_motion"),
         "masked video to video on ref2va: the subject replaced from a still and moving as the source's subject moved"),
        # The graph above with the frozen video cache on the song node's
        # model and nothing else changed (board card `use-frozen-row-cache`):
        # the first step of a window runs stock and later steps compute only
        # the regenerated rows and the text. A probe: no shipped or daily
        # graph carries the cache. Timed on one window, where it saved no
        # sampling time on this Sol-Attn chain
        # (bench/results/2026-10-06_frozen_cache_masked_window.md); the graph
        # stays as the arm that record ran.
        # The PDD8 freeze with the audio attention gain node in front of the
        # guider, inert as shipped; bench arms patch key_gain / value_gain.
        ("h3_candidate_t2v_pdd8_baked_audio_freeze_gain.json", "t2v-candidate-pdd8-baked-audio-freeze-gain",
         "t2v", LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler",
              unet=MODELS["unet_fl2va_pdd8_baked"],
              lora=(PDD_FL2VA_STRIPPED_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              freeze_audio=True, freeze_gain=True, out_prefix="Video/h3_candidate_t2v_pdd8_baked_audio_freeze_gain"),
         "CANDIDATE the PDD8 freeze with the audio attention gain knob, inert as shipped"),
        # The loop's first seam (plan step 6), API only: two windows of the
        # track, the second window's head frozen to the first window's tail
        # (39 frames, the smallest context on both clocks), decoded
        # separately, the overlap dropped, the two joined, the muxer on the
        # track's span. The seam is the thing to look at.
        ("h3_text_to_video_audio_freeze_2windows.json", "t2v-audio-freeze-2windows", "t2v",
         LONG_T2V_PROMPT,
         dict(freeze_windows=2, freeze_context=39, out_prefix="Video/h3_t2v_audio_freeze_2windows"),
         "text + a frozen audio track -> two windows joined at a 39-frame frozen seam"),
        ("h3_image_ref_plus_text_to_video.json", "r2v", "r2v", _ref_prompt(images=True), {},
         "reference image(s) + text -> video + audio"),
        ("h3_first_frame_to_video.json", "i2v", "i2v", None, {},
         "first frame + text -> video + audio (the conditioner takes its canvas from the keyframe)"),

        # fl2va: the same node, the same task string, one more LoadImage.
        # `last_frame` is what separates them, which is why the task stays
        # "i2v" -- inventing a fourth task value would fork the geometry and
        # canvas logic that both modes share exactly.
        ("h3_first_last_frame_to_video.json", "fl2v", "i2v", None,
         dict(last_frame=True, out_prefix="Video/h3_fl2v", **FL2V_CANVAS),
         "first frame + last frame + text -> video + audio"),

        # l2va: last frame ONLY, and the mode base_en names that this repo had
        # zero coverage of until 2026-08-28. `first_frame=False` is the whole
        # difference from fl2va -- the lone frame anchors the canvas rather
        # than being cropped into one chosen elsewhere, which the keyframe
        # comment in `build_api` already said was valid and which nothing
        # exercised. Task stays "i2v" for the same reason fl2va does: the
        # geometry and canvas logic are shared and a fourth task value would
        # fork them.
        #
        # It also exercises `l2v_prompt`, whose alignment sentence brackets
        # BOTH the label and the shot where fl2va brackets neither. That
        # distinction was carried by an uncalled branch emitting the guide's
        # placeholders literally until the same day.
        ("h3_last_frame_to_video.json", "l2v", "i2v", None,
         dict(last_frame=True, first_frame=False,
              out_prefix="Video/h3_l2v", **FL2V_CANVAS),
         "last frame + text -> video + audio (the closing frame is the anchor)"),

        # **The lightx2v turbo and SLA graphs were here and are retired as of
        # 2026-09-26** (owner: "lightx and turbo stuff irrelevant"); the list is
        # in docs/wiki/decisions.md under that date and the entries in git.

        # **The 2026-09-25 distill and audio-recovery probes** (docs/wiki/next_steps.md,
        # the 2026-09-25 blocks). Same prompt, seed, canvas and length as the
        # shipped t2v graphs, on the default chain, so each pairs with
        # h3_text_to_video_pdd and with the dense baseline at a matched seed.
        # C1, audio recovery: the shipped PDD8 graph plus an audio-only refine
        # pass on the undistilled model (audio_refine.py, h3_config.AUDIO_REFINE).
        ("h3_probe_t2v_pdd8_audio_refine.json", "t2v-pdd8-audio-refine", "t2v", LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler",
              lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS, audio_refine=True,
              refine_cache=True,
              out_prefix="Video/h3_probe_t2v_pdd8_audio_refine"),
         "text -> video + audio at 8 steps via PDD, then 6 undistilled audio-only steps"),

        # FlashGen (h3_config.FLASHGEN_*): a 4-step VSD distill as kijai converted
        # it, on its own four sigmas. Trained at 5.2 s; rendered at the owner's
        # length, which is a known confound.
        ("h3_probe_t2v_flashgen_4step.json", "t2v-flashgen-4step", "t2v", LONG_T2V_PROMPT,
         dict(lora=(FLASHGEN_LORA, FLASHGEN_STRENGTH), steps=FLASHGEN_STEPS,
              sampler_name=FLASHGEN_SAMPLER, manual_sigmas=FLASHGEN_MANUAL_SIGMAS,
              out_prefix="Video/h3_probe_t2v_flashgen_4step"),
         "text -> video + audio at 4 steps via the FlashGen LoRA on its own sigmas"),

        # FlashGen at the publisher's full rank 64 (FLASHGEN_R64_LORA), merged
        # by the stock loader: the control for h3_text_to_video_flashgen, which
        # applies it at the call (lora_branch.py). The pair isolates how the
        # LoRA reaches the int8 weight (bench/results/2026-09-26_int8_lora_requant.json).
        # The `_branch` probe that was the other half is gone since 0.147.1:
        # it was the shipped graph under another name.
        ("h3_probe_t2v_flashgen_r64_4step.json", "t2v-flashgen-r64-4step", "t2v", LONG_T2V_PROMPT,
         dict(lora=(FLASHGEN_R64_LORA, FLASHGEN_STRENGTH), lora_branch=False,
              steps=FLASHGEN_STEPS,
              sampler_name=FLASHGEN_SAMPLER, manual_sigmas=FLASHGEN_MANUAL_SIGMAS,
              out_prefix="Video/h3_probe_t2v_flashgen_r64_4step"),
         "FlashGen at full rank 64, merged by LoraLoaderModelOnly"),
        # The same with Sol off, on the kitchen backend: vllm-omni's NPU recipe
        # runs FlashGen dense with the sparse config dropped.
        ("h3_probe_t2v_flashgen_r64_4step_branch_dense.json", "t2v-flashgen-r64-4step-branch-dense",
         "t2v", LONG_T2V_PROMPT,
         dict(lora=(FLASHGEN_R64_LORA, FLASHGEN_STRENGTH), lora_branch=True,
              dense_attn="ck", sol_on=False,
              steps=FLASHGEN_STEPS, sampler_name=FLASHGEN_SAMPLER,
              manual_sigmas=FLASHGEN_MANUAL_SIGMAS,
              out_prefix="Video/h3_probe_t2v_flashgen_r64_4step_branch_dense"),
         "FlashGen rank 64 at the call, dense attention as vllm-omni's recipe runs it"),

        # FastH3 8-step V2 as ComfyUI's own template runs it (h3_config.FASTH3_*):
        # kitchen backend, core's VSA in Sol's slot, Sol off because VSA replaces
        # the block attention it would override. T2VA only.
        ("h3_probe_t2v_fasth3_8step.json", "t2v-fasth3-8step", "t2v", LONG_T2V_PROMPT,
         dict(dense_attn="ck", sol_on=False, unet=MODELS["unet_fasth3_v2"],
              steps=FASTH3_STEPS, sampler_name=FASTH3_SAMPLER,
              scheduler_name=FASTH3_SCHEDULER, shift=FASTH3_SHIFT,
              core_vsa=FASTH3_CORE_VSA,
              out_prefix="Video/h3_probe_t2v_fasth3_8step"),
         "text -> video + audio at 8 steps via FastVideo's FastH3 V2, core VSA"),

        # FastH3 V2 to FastVideo's own contract (h3_config.FASTH3_CONTRACT_*),
        # and the two halves of the change on their own: the attention (keep
        # 20%, VSA from the first step) and the sampling (the release's
        # positions on Euler). The template graph above is the fourth arm.
        ("h3_probe_t2v_fasth3_8step_contract.json", "t2v-fasth3-8step-contract", "t2v", LONG_T2V_PROMPT,
         dict(dense_attn="ck", sol_on=False, unet=MODELS["unet_fasth3_v2"],
              steps=FASTH3_STEPS, sampler_name=FASTH3_CONTRACT_SAMPLER,
              manual_sigmas=FASTH3_CONTRACT_SIGMAS, shift=FASTH3_SHIFT,
              core_vsa=FASTH3_CONTRACT_VSA,
              out_prefix="Video/h3_probe_t2v_fasth3_8step_contract"),
         "FastH3 V2 as FastVideo's contract runs it: its sigmas on Euler, VSA keep 20% on every step"),
        ("h3_probe_t2v_fasth3_8step_contract_attn.json", "t2v-fasth3-8step-contract-attn", "t2v", LONG_T2V_PROMPT,
         dict(dense_attn="ck", sol_on=False, unet=MODELS["unet_fasth3_v2"],
              steps=FASTH3_STEPS, sampler_name=FASTH3_SAMPLER,
              scheduler_name=FASTH3_SCHEDULER, shift=FASTH3_SHIFT,
              core_vsa=FASTH3_CONTRACT_VSA,
              out_prefix="Video/h3_probe_t2v_fasth3_8step_contract_attn"),
         "FastH3 V2 on the template's sampling with the contract's attention only"),
        ("h3_probe_t2v_fasth3_8step_contract_sampling.json", "t2v-fasth3-8step-contract-sampling", "t2v",
         LONG_T2V_PROMPT,
         dict(dense_attn="ck", sol_on=False, unet=MODELS["unet_fasth3_v2"],
              steps=FASTH3_STEPS, sampler_name=FASTH3_CONTRACT_SAMPLER,
              manual_sigmas=FASTH3_CONTRACT_SIGMAS, shift=FASTH3_SHIFT,
              core_vsa=FASTH3_CORE_VSA,
              out_prefix="Video/h3_probe_t2v_fasth3_8step_contract_sampling"),
         "FastH3 V2 on the contract's sampling with the template's attention only"),

        ("h3_probe_t2v_flashgen_4step_audio_refine.json", "t2v-flashgen-4step-audio-refine", "t2v",
         LONG_T2V_PROMPT,
         dict(lora=(FLASHGEN_R64_LORA, FLASHGEN_STRENGTH), lora_branch=True, steps=FLASHGEN_STEPS,
              sampler_name=FLASHGEN_SAMPLER, manual_sigmas=FLASHGEN_MANUAL_SIGMAS,
              audio_refine=True, refine_cache=True,
              out_prefix="Video/h3_probe_t2v_flashgen_4step_audio_refine"),
         "the FlashGen arm plus 6 undistilled audio-only steps, the frozen video cached"),

        # The cache's control: the same arm with the refine pass uncached.
        # The refine graphs carry the cache since 0.144.0, after the owner
        # heard no difference (bench/results/2026-09-25_frozen_cache_s1.md).
        ("h3_probe_t2v_flashgen_4step_audio_refine_uncached.json",
         "t2v-flashgen-4step-audio-refine-uncached", "t2v", LONG_T2V_PROMPT,
         dict(lora=(FLASHGEN_R64_LORA, FLASHGEN_STRENGTH), lora_branch=True, steps=FLASHGEN_STEPS,
              sampler_name=FLASHGEN_SAMPLER, manual_sigmas=FLASHGEN_MANUAL_SIGMAS,
              audio_refine=True,
              out_prefix="Video/h3_probe_t2v_flashgen_4step_audio_refine_uncached"),
         "the FlashGen refine arm without the frozen-video cache, its control"),

        # First graph in this repo to wire a reference VIDEO. Everything about
        # that path was read off source until 2026-08-13 and never executed.
        # The reference-combination matrix. Five arms, one per shape of
        # ref2va request, each with a prompt that declares EXACTLY the labels
        # its own graph wires -- `bench/check_ref_prompt_labels.py` enforces
        # that agreement, because the tokenizer derives the labels from the
        # sockets and a prompt naming one that is not there fails silently.
        ("h3_ref_video_to_video.json", "r2v-video", "r2v",
         _ref_prompt(images=True, video=True, video_audio=True),
         dict(**REF_VIDEO_BUDGET, ref_video=True, out_prefix="Video/h3_r2v_video"),
         "images + reference video + its soundtrack -> video + audio"),

        ("h3_ref_video_only.json", "r2v-video-only", "r2v",
         _ref_prompt(images=False, video=True),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_video_audio=False, ref_images_on=False,
              out_prefix="Video/h3_r2v_video_only"),
         "reference video only, silent clip"),

        ("h3_ref_video_audio.json", "r2v-video-audio", "r2v",
         _ref_prompt(images=False, video=True, video_audio=True),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_images_on=False,
              out_prefix="Video/h3_r2v_video_audio"),
         "reference video + its soundtrack, no images"),

        # One named two-stage policy, not two independently switchable fixes.
        # The release first puts the full-rate clip on the reference canvas for
        # the VAE, then independently applies its duration-aware processor to
        # the raw 2 fps Qwen samples. Upscale alone would overshoot the long-
        # clip Qwen budget, so the probe toggles both at one compiler boundary.
        ("h3_probe_release_video_policy.json", "r2v-video-release-policy", "r2v",
         _ref_prompt(images=False, video=True, video_audio=True),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_images_on=False,
              ref_video_policy="release",
              out_prefix="Video/h3_probe_release_video_policy"),
         "reference video on the atomic release VAE/Qwen preparation policy"),

        ("h3_ref_image_audio.json", "r2v-image-audio", "r2v",
         _ref_prompt(images=True, audio=True),
         dict(ref_audio=True, out_prefix="Video/h3_r2v_image_audio"),
         "reference images + standalone audio"),

        ("h3_ref_image_video_audio.json", "r2v-all", "r2v",
         _ref_prompt(images=True, video=True, video_audio=True, audio=True),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_audio=True,
              out_prefix="Video/h3_r2v_all"),
         "images + video + its soundtrack + standalone audio"),

        # A CAPTURE target, not a render target. It exists so `h3_capture.py`
        # has a graph to point at, and so the next capture is a re-run rather
        # than archaeology: the 2026-08-15 capture's config survives only as
        # prose in a README on the share, which is why nothing about it could
        # be reproduced from this repo.
        #
        # Sol-Attn is off on this graph, and that is a REQUIREMENT
        # here rather than the usual default. Dense attention is
        # permutation-equivariant, so one capture serves every token ordering
        # at every block; a capture taken with Sol on is a slice of a
        # trajectory that diverged, and only valid for the arm that took it.
        #
        # 362 frames at 1024x768 with native reference sizes is roughly 97k
        # tokens: above the ~60k floor `docs/SOLATTN.md` warns about, and well
        # under the 182,092 that OOMed this card. All three numbers come from
        # REF_VIDEO_BUDGET, so they move with the other reference arms.
        ("h3_probe_capture_ref3.json", "probe-capture-ref3", "r2v",
         _ref_prompt(images=("character", "garment", "environment")),
         # dense_attn="sage" (sol_on=False until 2026-09-15, when the default
         # floor stopped being sage): Sol absent and sage KEPT, because
         # `h3_capture.py` records from inside the sage forward. And this is
         # the ONE graph that earns the exception to the Sol-on-by-default rule: `h3_capture.py` records the activations a
         # dense baseline is measured from, and a Sol arm gives sage only a
         # subset of the sampler's steps, so a capture taken through Sol is a
         # different trajectory than the one the analysis assumes.
         #
         # It was previously done by hand-editing the emitted `_api.json` to
         # route past the Sol node, which CLAUDE.md forbids and which the next
         # regeneration silently reverted -- putting Sol back into the capture
         # chain with nothing going red. Declaring it here survives regeneration.
         dict(**REF_VIDEO_BUDGET, ref_images=CAPTURE_REF_IMAGES,
              dense_attn="sage",
              out_prefix="Video/h3_probe_capture_ref3"),
         "3 references spanning 0.78-4.23 MP; the h3_capture.py target"),

        # The VAE encoder-precision arms. ComfyUI carries ONE dtype per VAE and
        # `--fp32-vae` moves both halves; the release keeps this VAE resident in
        # fp32 *because the same module encodes references and keyframes* and
        # decodes under fp16, which the flag cannot express.
        # `MiniMaxH3VAEPrecision` splits them, and until 2026-08-21 it was wired
        # into zero graphs -- a fix nobody can reach. The node that named this
        # failure mode, `MiniMaxH3VendorTokens`, was removed on 2026-08-27
        # after ComfyUI made it a no-op; the shape it illustrated is what
        # matters, not the example.
        #
        # **These two arms cannot answer which output is better**, and building
        # them as though they could would be the mistake CLAUDE.md records: a
        # rendered clip cannot A/B a numerical change, on any sampler. They
        # price it -- VRAM, time, and that it runs end to end in a real graph --
        # and they give a seed sweep something to sweep if anyone ever wants a
        # distribution. The controlled comparison is at the call, in
        # `bench/grade_vae_encoder_precision.py`.
        *[
            (f"h3_probe_ref_vae_encoder_{tag}.json", f"r2v-vaeenc-{tag}", "r2v",
             _ref_prompt(images=("character", "garment", "environment")),
             dict(**REF_VIDEO_BUDGET, ref_images=CAPTURE_REF_IMAGES,
                  vae_encoder=enc,
                  out_prefix=f"Video/h3_probe_ref_vae_encoder_{tag}"),
             note)
            for tag, enc, note in (
                ("fp16", None,
                 "the stock arm: one dtype for the whole VAE, whatever "
                 "ComfyUI resolved. The baseline the fp32 arm is priced "
                 "against, and it wires no precision node at all"),
                ("fp32", "fp32",
                 "the encoder promoted to fp32, matching the release's "
                 "residency for the half that encodes references. ~0.34 GiB "
                 "on the shipped checkpoint; the decoder is left alone"),
            )
        ],


        # The same capture on the fl2va checkpoint with no LoRA: the control
        # the block-49 attribution was missing. The 2026-08-17 capture was
        # fl2va + the ref LoRA and the 2026-08-18 one ref2va, so "the loud
        # heads are a property of the released weights, not of ref2va" rested
        # on the gains matching between checkpoints rather than on a clean
        # fl2va observation. Same prompt, references, canvas, length and seed
        # as the twin above, so the only thing that differs is the unet. fl2va
        # was not trained on reference rows; that is fine for a capture, whose
        # question is what the attention inputs look like, not whether the
        # clip is good. Sol off for the same reason as the twin.
        ("h3_probe_capture_ref3_fl2va.json", "probe-capture-ref3-fl2va", "r2v",
         _ref_prompt(images=("character", "garment", "environment")),
         dict(**REF_VIDEO_BUDGET, ref_images=CAPTURE_REF_IMAGES,
              dense_attn="sage", unet=MODELS["unet_fl2va"],
              out_prefix="Video/h3_probe_capture_ref3_fl2va"),
         "the capture twin on fl2va with no LoRA; the missing block-49 control"),

        # The reference-view ablation, second edition (2026-09-13). One
        # ref2va graph per scene in `h3_config.REFVIEW2_SCENES`, built at
        # the node defaults -- vendor parity: every still to the 2048 short
        # edge, upscale on, one copy for both towers -- and the arms are
        # widget patches in `bench/refview2_arms.json`, applied by
        # `run_graph_arms.py --manifest`, so five graphs carry thirty arms.
        # Shipped defaults otherwise: Sol on, sage on, the base step count,
        # no PDD. `docs/h3_references.md` is why the two copies of a still
        # need not share a geometry and what each arm asks.
        *[
            (f"h3_probe_refview2_{tag}.json", f"r2v-refview2-{tag}", "r2v",
             _bank_prompt(prompt_id),
             dict(length=LONG_LENGTH, ref_image_count=len(stills),
                  ref_images=stills,
                  # Pinned: "as built" is the manifest's `parity` arm, and it
                  # means the 2026-09-13 values, not whatever the node
                  # default is now.
                  ref_qwen_short_edge=0,
                  out_prefix=f"Video/h3_probe_refview2_{tag}"),
             f"reference-view ablation scene: {prompt_id}")
            for tag, prompt_id, stills in REFVIEW2_SCENES
        ],

        # Reference pathway ablation, 2026-09-03. ComfyUI PR 16065 (core
        # commit 1aec3a13) made both VAE inputs on MiniMaxH3ReferenceToVideo
        # optional: with no VAE a reference is presented to Qwen3-VL exactly
        # as before (same label, same vision block, the same video-modality
        # tag on its text span) and adds no reference-latent rows to the
        # DiT, so `minimax_refs` is never set and the DiT lays the sequence
        # out as it would for a text-only pass. Our conditioner mirrors that
        # since 0.99.33. Five arms, one prompt, one seed, the capture stills:
        #
        #   typed_both      ours, ref2va, both pathways -- the shipped path
        #   typed_encoder   ours, ref2va, encoder only (no VAE wired)
        #   native_both     core's node, ref2va, both pathways
        #   native_encoder  core's node, ref2va, encoder only
        #   fl2va_encoder   ours, the fl2va checkpoint, encoder only. That
        #                   partition meets <Picture> blocks as keyframes, so
        #                   this asks whether it reads them as identity hints
        #                   when no keyframe rows follow. Off its trained
        #                   structure (three pictures, no keyframe); a bound,
        #                   not a shipped call.
        #
        # typed against native is NOT a same-footing pair: core sizes each
        # still by `ref_image_size=match` and feeds one tensor to both
        # consumers; ours sizes by `size_policy=max` with a separate Qwen
        # view. The controlled comparisons are within a family (both against
        # encoder) and between the two encoder-only arms, where no VAE view
        # exists to differ. `bench/ref_pathway_arms.json` is the manifest.
        *[
            (f"h3_probe_ref_pathway_{tag}.json", f"r2v-pathway-{tag}", "r2v",
             _ref_prompt(images=("character", "garment", "environment")),
             dict(**REF_VIDEO_BUDGET, ref_images=CAPTURE_REF_IMAGES,
                  ref_latents=latents, **more,
                  out_prefix=f"Video/h3_probe_ref_pathway_{tag}"),
             what)
            for tag, latents, more, what, note in (
                ("typed_both", True, {},
                 "reference pathway arm: our conditioner, ref2va, encoder and DiT rows",
                 "Reference pathway ablation. Our conditioner with both VAEs "
                 "wired: every still reaches Qwen3-VL and the DiT gets its "
                 "reference-latent rows. The shipped behaviour, and the "
                 "control for `typed_encoder`."),
                ("typed_encoder", False, {},
                 "reference pathway arm: our conditioner, ref2va, encoder only",
                 "Reference pathway ablation. Our conditioner with neither VAE "
                 "wired: the same stills, labels and Qwen view, and no "
                 "reference rows in the DiT. Judged blind against "
                 "`typed_both` on matched seeds; what survives is what the "
                 "encoder pathway carries on its own."),
                ("native_both", True, dict(native_ref=True),
                 "reference pathway arm: core's node, ref2va, encoder and DiT rows",
                 None),
                ("native_encoder", False, dict(native_ref=True),
                 "reference pathway arm: core's node, ref2va, encoder only",
                 None),
                ("fl2va_encoder", False, dict(unet=MODELS["unet_fl2va"]),
                 "reference pathway arm: our conditioner on fl2va, encoder only",
                 "Reference pathway ablation on the fl2va checkpoint. Neither "
                 "VAE wired, so the stills reach the DiT only through the "
                 "encoder's vision tokens, which is the one form of picture "
                 "this partition was trained to read. Three pictures and no "
                 "keyframe is off its trained structure; read the result as "
                 "a bound on what the encoder pathway can do, not as a "
                 "recipe."),
            )
        ],

        ("h3_ref_video_edit.json", "r2v-edit", "r2v",
         _ref_prompt(images=False, video=True, video_audio=True, video_role="edit"),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_images_on=False,
              out_prefix="Video/h3_r2v_edit"),
         "edit a source video -- the closest thing H3 has to inpainting"),

        ("h3_ref_video_image_edit.json", "r2v-edit-combo", "r2v",
         _ref_prompt(images=True, video=True, video_audio=True, video_role="edit"),
         dict(**REF_VIDEO_BUDGET, ref_video=True,
              out_prefix="Video/h3_r2v_edit_combo"),
         "edit a source video, with images supplying what replaces what"),

        # The twin of h3_ref_video_image_edit: same sockets, same budget, a
        # different request. Kept adjacent so the pair reads as the A/B it is.
        ("h3_ref_video_swap.json", "r2v-swap", "r2v",
         _ref_prompt(images=True, video=True, video_audio=True,
                     video_role="swap", audio_role="copy"),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_image_count=1,
              out_prefix="Video/h3_r2v_swap"),
         "replace a character in a source video with one from an image"),

        ("h3_ref_video_continue.json", "r2v-continue", "r2v",
         _ref_prompt(images=False, video=True, video_audio=True, video_role="continue"),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_images_on=False,
              out_prefix="Video/h3_r2v_continue"),
         "continue from the end of a source video"),

        ("h3_ref_video_motion.json", "r2v-motion", "r2v",
         _ref_prompt(images=True, video=True, video_role="motion"),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_video_audio=False,
              out_prefix="Video/h3_r2v_motion"),
         "transfer motion from a video onto a subject from an image"),

        ("h3_ref_audio_voice.json", "r2v-voice", "r2v",
         _ref_prompt(images=True, audio=True, audio_role="voice"),
         dict(ref_audio=True, out_prefix="Video/h3_r2v_voice"),
         "reference a speaker's voice timbre for generated speech"),

        # --- probes: pairs, one variable, run against the named twin ---

        # The equal-cost shape control. 21:9, 16:9 and 9:16 are all
        # (w//32)*(h//32) = 1008 tokens/frame, so all three run at the SAME
        # sequence length and the same attention cost while the long edge goes
        # 768 -> 1536. Every other probe here changes cost to change shape;
        # these two change shape with cost held exactly constant, which is the
        # only way to ask whether the model is actually shape-neutral.
        ("h3_probe_canvas_ultrawide.json", "t2v-21by9", "t2v", LONG_T2V_PROMPT,
         dict(width=1536, height=672, out_prefix="Video/h3_probe_21by9"),
         "21:9, the same cost as the default canvas"),

        ("h3_probe_canvas_portrait.json", "t2v-9by16", "t2v", LONG_T2V_PROMPT,
         dict(width=768, height=1344, out_prefix="Video/h3_probe_9by16"),
         "9:16 portrait, the same cost as the default canvas"),

        # --- Parallel Decoding Distillation -----------------------------
        # Not a step distillation. The trajectory stays a 32-point grid; the
        # final output head is replicated per interval and each step decodes a
        # block of four as one mean velocity, so 8 evaluations cover it. See
        # docs/h3_pdd.md.
        #
        # The shift does NOT move, and that is the point: the block boundaries
        # are the plain 8-step shifted schedule bit for bit, so these graphs
        # differ from a base ref2va arm in the loader and the step count and in
        # nothing else. Every other accelerator here moves at least two things.
        #
        # EULER. Since 2026-08-27 that is the default for every distilled arm
        # (h3_config.DISTILL_SAMPLING), but PDD required it before the policy
        # existed and would require it if the policy changed: alibaba-pai's own
        # scheduler takes "one Euler (eta = 0)
        # step" and their adapter defines the fused head as "the mean velocity
        # of one block, which an Euler step over the block boundaries
        # consumes". er_sde injects noise and uses a different update rule, so
        # the heads would be consumed by something they were not distilled
        # against.
        #
        # The scheduler stays `simple`, which bench/check_distill_grid.py
        # measures as EXACT at 4 and 8 steps -- it reads the discrete
        # 1,000-entry table and both divide 1,000 -- so these graphs already
        # sample precisely on the boundaries the heads were fused at. That is
        # now graded per graph against pdd_math.block_bounds rather than
        # assumed.
        #
        # NO DiT SELF-ATTENTION PATCHING on any of these: no sage, no Sol.
        # A DiT-side statement only -- the Qwen3-VL encoder resolves its own
        # attention inside its decoder forward and neither node reaches it,
        # since both take io.Model.Input. Both DiT patches change
        # attention numerics, and the subject of every arm here is a numerical
        # mechanism in the output head. Leaving them wired puts two
        # approximations in the path of an experiment about a third, and the
        # head-selection defect of 2026-08-26 is exactly the kind of thing they
        # would have made unattributable. Owner decision, 2026-08-26.
        #
        # It is also what the reference runs: their pipeline is Diffusers'
        # ModularPipeline on stock SDPA, so dense here IS the vendor
        # configuration rather than a handicap. It costs about 2.4x on this
        # workload -- 70.3 s/it against 28.7 -- because at ~90k packed tokens
        # attention is quadratic and dominates. That price is for replication;
        # it is not the configuration to render production clips in.
        #
        # Trained on transformer_ref itself, so docs/h3_ref2v_distillation.md's
        # Fact B -- an fl2v distill aimed at the wrong weights -- does not
        # apply to this one. Facts A and C do not follow from that and are
        # untouched.

        # --- the market scene as ref2va, base and both PDD step counts ------
        # The t2v market rewrite (d5be353) was rendered at 16 steps on
        # 2026-08-28 and the owner's verdict was that it fixed the scene. These
        # ask the next question: does the SAME scene hold up as a reference
        # task, and does it survive distillation at 8 and at 4 evaluations.
        #
        # Matched to the t2v arms deliberately -- same canvas, same length, same
        # seed, same shift -- so the only intended differences are the task, the
        # reference, and the step count. Two differences are NOT free variables
        # and must not be normalised away: PDD requires `euler` (a fused head is
        # the block's mean velocity and one Euler step integrates exactly that),
        # and Sol's `end_percent` was step-aware and derived until
        # 2026-09-11 (1.0 on every graph since).
        #
        # The reference is a runway photograph, deliberately far from the role
        # it is being asked to fill. That is the owner's choice and it makes the
        # arm a harder identity test than a plausible-looking stallholder would.
        #
        # **MEMORY, 2026-08-28. Marginal and ORDER-DEPENDENT, not a ceiling.**
        # `h3_ref2v_market_pdd` OOMed once at 17.5 MiB free and then SUCCEEDED
        # on retry with nothing changed. The full sequence is the evidence:
        #
        #   r2v16     no LoRA                        success
        #   r2v_pdd8  PDD, straight after r2v16      OOM, 17.5 MiB free
        #   r2v_pdd4  PDD, after that OOM            success
        #   r2v_pdd8  PDD, retry                     SUCCESS
        #
        # Same graph, same card. So "PDD ref2va does not fit at this canvas and
        # length" is REFUTED, and an earlier version of this note said it. The
        # tell was there before the retry: short by 17.5 MiB is short by
        # nothing, and the 4-step had already passed at an identical profile.
        #
        # **The mechanism, and it is not the one everybody reaches for.** The
        # failing attempt was the first to apply the PDD LoRA on top of a model
        # loaded WITHOUT it; the passing ones ran when a PDD-patched model was
        # already the resident shape. The peak is in the TRANSITION, not the
        # steady state, which is exactly why it is sensitive to what ran before.
        #
        # And it is not the head bank. The artifact is 1,059 MiB, of which the
        # 32-head bank is 42 MiB (4%) and the backbone rank-64 LoRA A/B pairs
        # are 933 MiB (88%), applied through ComfyUI's native `add_patches`.
        # This note previously blamed "the resident head bank", which was the
        # available explanation rather than the measured one.
        #
        # What survived both corrections: `h3_ref2v_market` succeeded on the
        # same scene, reference and canvas, so whatever this is, it is not the
        # scene and not the reference.
        # **The two scene arms, and the only graphs here that carry a marker
        # other than `<d>`.** `REF_SCENE_SHOTS` and `REF_SCENE_AUDIO` have held
        # both scenes since they were written, `_ref_prompt(scene=...)` renders
        # them, and until 2026-08-28 NO CALL SITE PASSED `scene` -- so
        # `<|caption_start|>`, `<|caption_end|>`, `<|lyrics_start|>` and
        # `<|lyrics_end|>` appeared in this generator and in zero shipped
        # graphs. Wiring them is what makes the marker path reachable at all.
        #
        # They close a second gap at the same time: `docs/prompting.md` 9.10
        # records the guide's 350-500 word budget for a generation
        # `detailed_description`, and these are written inside it (the current
        # counts are `bench/build_prompt_bank.py`'s, not a number here). They have
        # since been rendered, and `docs/scene_arm_renders.md` reads those renders;
        # the marker coverage is a property of the TEXT and not a claim about what
        # the model does with it.
        *[
            (f"h3_ref2v_scene_{sc}.json", f"r2v-scene-{sc}", "r2v",
             _ref_prompt(images=("character",), scene=sc),
             dict(ref_images=SCENE_REF_IMAGES[sc], ref_image_count=1,
                  out_prefix=f"Video/h3_r2v_scene_{sc}"),
             note)
            for sc, note in (
                ("subway", "a busker on a crowded platform, four shots with "
                           "sung lyrics and a platform sign -- the lyrics and "
                           "caption marker arm"),
                ("kitchen", "a restaurant pass mid-service, four shots with "
                            "overlapping dialogue and a ticket caption -- the "
                            "same markers against speech rather than song"))
        ],

        ("h3_ref2v_market.json", "r2v-market", "r2v", MARKET_REF2V_PROMPT,
         dict(ref_images=MARKET_REF_IMAGES, ref_image_count=1,
              out_prefix="Video/h3_r2v_market"),
         "the market scene as ref2va, base 16 steps"),

        ("h3_ref2v_market_pdd.json", "r2v-market-pdd8", "r2v", MARKET_REF2V_PROMPT,
         dict(ref_images=MARKET_REF_IMAGES, ref_image_count=1,
              pdd=True, sampler_name="euler",
              lora=(PDD_REF2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              out_prefix="Video/h3_r2v_market_pdd"),
         "the market scene as ref2va at 8 steps via PDD"),

        ("h3_ref2v_market_pdd_4step.json", "r2v-market-pdd4", "r2v", MARKET_REF2V_PROMPT,
         dict(ref_images=MARKET_REF_IMAGES, ref_image_count=1,
              pdd=True, sampler_name="euler",
              lora=(PDD_REF2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS_FAST,
              out_prefix="Video/h3_r2v_market_pdd_4step"),
         "the market scene as ref2va at 4 steps via PDD"),

        # --- the dialogue probe, t2v and ref2v -----------------------------
        # Eight short lines over three shots with the pacing written in. The
        # pair exists because they differ on exactly one axis: where the two
        # people come from. Everything judgeable -- the lines, the cut times,
        # the speaker order, the soundscape -- is identical, so anything that
        # separates them is the reference conditioning and not the script.
        #
        # 362 frames, which is the longest length H3 was trained on and the
        # last shot needs it.
        # **These three carried `length=362` as a literal until 2026-08-30**,
        # so the owner's move of the long default to 345 did not reach them and
        # they were the only graphs left off the audio clock. Now on
        # LONG_LENGTH like everything else.
        #
        # Safe for their prompts, checked rather than assumed on 2026-08-30,
        # when the shot headers carried times (00:06.000 and 00:11.000, inside
        # 345 frames as they were inside 362); they carry none now
        # (`docs/prompting.md` section 3.1). What changes is the final shot, from
        # 4.08s to 3.38s, and its direction is "holds still until the final
        # frame", which does not name a duration.
        ("h3_text_to_video_dialogue.json", "t2v-dialogue", "t2v",
         DIALOGUE_T2V_PROMPT,
         dict(length=LONG_LENGTH, out_prefix="Video/h3_t2v_dialogue"),
         "two speakers, eight clipped lines, dialogue markers throughout"),

        ("h3_image_ref_plus_text_to_video_dialogue.json", "r2v-dialogue", "r2v",
         DIALOGUE_REF2V_PROMPT,
         dict(length=LONG_LENGTH, ref_image_count=2, ref_images=DIALOGUE_REF_IMAGES,
              out_prefix="Video/h3_r2v_dialogue"),
         "the same exchange, both speakers from reference stills"),

        # The triple. Every PAIR of {references, dialogue, distill} shipped and
        # the combination did not, which is why nothing here could reproduce
        # the 2026-08-27 misattribution -- that needed all three.
        #
        # Read against the prompt rather than against a twin. The subject
        # definitions bind <Subject 1> to <Picture 2> and <Subject 2> to
        # <Picture 1> BY NUMBER over eight `<d>` lines, so "which mouth said
        # which line" is a fact about the render. That is what makes this one
        # of the few arms here that does not need a distribution: CLAUDE.md's
        # different-sample rule bites when two clips are compared, and nothing
        # is being compared.
        ("h3_image_ref_plus_text_to_video_dialogue_pdd_4step.json",
         "r2v-dialogue-pdd4", "r2v", DIALOGUE_REF2V_PROMPT,
         dict(pdd=True, sampler_name="euler",
              lora=(PDD_REF2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS_FAST,
              length=LONG_LENGTH, ref_image_count=2, ref_images=DIALOGUE_REF_IMAGES,
              out_prefix="Video/h3_r2v_dialogue_pdd_4step",
              ),
         "the same stairwell exchange from two stills, at 4 steps via PDD"),


        # --- PDD, the arms to actually render with ------------------------
        # The repo default: a dense kernel AND Sol (sage until 2026-09-15, the
        # kitchen backend node since; `_attention_plan`).
        #
        # These carried sage only until 2026-08-26, on the reasoning that Sol's
        # adaptive per-step skipping is incoherent against a fixed fused block
        # schedule. **That was wrong and the numbers say so.** Sol's window is a
        # PERCENT band (start 0.2, end 0.9) resolved off the sigma curve, so its
        # coverage scales with the step count instead of degrading: 11 of 16
        # steps, 6 of 8, 3 of 4 -- computed against the real curve at shift 12,
        # and the 16-step figure reproduces the 11-sparse/5-dense this repo had
        # already measured. Sol and PDD also touch different surfaces entirely,
        # attention against the output head, and `ModelPatcher.clone` carries
        # object patches, so a Sol node downstream keeps PDD's three.
        #
        # sage alone buys ~2.4x here, which at ~90k packed tokens is worth more
        # than halving the steps; Sol is on top of that.
        #
        # Their dense twins under h3_probe_ref2v_pdd* are the reference
        # configuration -- Diffusers' stock SDPA, what the vendor runs -- and
        # exist to be compared against, not rendered with.
        ("h3_text_to_video_pdd.json", "texttovideopdd", "t2v", LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler",
              lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              out_prefix="Video/text_to_video_pdd"),
         "text -> video + audio at 8 steps via PDD, kitchen dense + Sol"),
        # The control for the shipped graph above since 0.154.0: PDD's backbone
        # merged into the int8 weight by ComfyUI's requantising merge, as every
        # PDD graph did before. Only `backbone_apply` differs.
        ("h3_probe_t2v_pdd8_merge.json", "t2v-pdd8-merge", "t2v", LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler",
              lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              pdd_backbone_apply="merge",
              out_prefix="Video/h3_probe_t2v_pdd8_merge"),
         "PDD8 with its backbone merged into int8 (the pre-0.154.0 path), the branch's control"),

        # **The PDD ladder's own rungs, added 2026-09-04.** The 2026-09-03
        # speedup ladder rendered PDD8 only as the shipped graph (sage plus
        # Sol at the PDD window) and it lost to the true baseline on every
        # scene; the owner pointed out that no arm rendered PDD8 without Sol
        # or without sage, so the loss attributes to nothing narrower than
        # the shipped graph. These two are the missing rungs. Their prompts
        # are patched per scene from the bank by bench/pdd_ladder_arms.json.

        ("h3_probe_t2v_pdd8_sage.json", "t2v-pdd8-sage", "t2v", LONG_T2V_PROMPT,
         dict(pdd=True, dense_attn="sage", sampler_name="euler",
              lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              out_prefix="Video/h3_probe_t2v_pdd8_sage"),
         "text -> video + audio at 8 steps via PDD, sage alone, Sol absent"),

        ("h3_probe_t2v_pdd8_dense.json", "t2v-pdd8-dense", "t2v", LONG_T2V_PROMPT,
         dict(pdd=True, dense_attn=True, sampler_name="euler",
              lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              out_prefix="Video/h3_probe_t2v_pdd8_dense"),
         "text -> video + audio at 8 steps via PDD, stock attention, the PDD ladder's baseline"),

        # **The bake pair's arm, added 2026-09-05.** Identical to
        # h3_probe_t2v_pdd8_sage except the checkpoint and the sidecar: the
        # PDD backbone is folded into the int8 weights offline
        # (bench/bake_pdd_checkpoint.py) and the sidecar carries only what the
        # bake cannot hold. Same steps, same kernels, sage alone, so the pair
        # differs in one thing: whether the backbone delta reaches the weights
        # through a requantised merge at load or was quantised once with them.
        ("h3_probe_t2v_pdd8_baked_sage.json", "t2v-pdd8-baked-sage", "t2v", LONG_T2V_PROMPT,
         dict(pdd=True, dense_attn="sage", sampler_name="euler",
              unet=MODELS["unet_fl2va_pdd8_baked"],
              lora=(PDD_FL2VA_STRIPPED_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              out_prefix="Video/h3_probe_t2v_pdd8_baked_sage"),
         "text -> video + audio at 8 steps via PDD on the baked checkpoint, sage alone"),

        # **The description-length pair.** Same PDD 4-step settings as the other
        # t2v PDD arms, so length is the only thing that differs from each
        # other AND the configuration is the one artifacts show up in. See the
        # constants for what is held (word counts: `docs/prompt_bank.md`), with
        # identical dialogue, camera moves, cuts and shot structure.
        *[
            (f"h3_text_to_video_aisle_{tag}.json", f"t2v-aisle-{tag}", "t2v",
             prompt,
             dict(pdd=True, sampler_name="euler",
                  lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS_FAST,
                  out_prefix=f"Video/h3_t2v_aisle_{tag}"),
             note)
            for tag, prompt, note in (
                ("short", T2V_AISLE_SHORT,
                 "the hardware aisle at low demand -- control arm of the "
                 "demand pair"),
                ("long", T2V_AISLE_LONG,
                 "the same scene with more fine structure demanded -- no new "
                 "subject, action, camera move or line"))
        ],

        # **The predictability pair.** One shot each, no cuts, both long, so
        # shot count and length are equalised and only how predictable the
        # next frame is varies. Delta says the rail arm is WORST; the owner's
        # reading says it is clean. They cannot both be right.
        *[
            (f"h3_text_to_video_{tag}_long.json", f"t2v-{tag}-long", "t2v",
             prompt,
             dict(pdd=True, sampler_name="euler",
                  lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS_FAST,
                  out_prefix=f"Video/h3_t2v_{tag}_long"),
             note)
            for tag, prompt, note in (
                ("rail", T2V_RAIL_LONG,
                 "one rail move across a boxy house, nothing else moving -- "
                 "maximal delta, minimal uncertainty"),
                ("churn", T2V_CHURN_LONG,
                 "one handheld take through a night market under changing "
                 "LEDs -- similar delta, no extrapolable structure"))
        ],

        # The second pair, different scene, same manipulation. Two independent
        # scenes is what separates a length effect from a fact about one scene.
        *[
            (f"h3_text_to_video_sortline_{tag}.json", f"t2v-sortline-{tag}",
             "t2v", prompt,
             dict(pdd=True, sampler_name="euler",
                  lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS_FAST,
                  out_prefix=f"Video/h3_t2v_sortline_{tag}"),
             note)
            for tag, prompt, note in (
                ("short", T2V_SORTLINE_SHORT,
                 "the sorting line at normal length -- control arm of the "
                 "second description-length pair"),
                ("long", T2V_SORTLINE_LONG,
                 "the same sorting line elaborated, no new content"))
        ],

        # --- FlashGen, the 4-step arm to render with (2026-09-26) ----------
        # The owner: "lets take our best stab at making a good workflow ship".
        # Every choice and its source is docs/research/2026-09-26_flashgen.md:
        # the publisher's full rank 64 (FLASHGEN_R64_LORA) applied at the call
        # (lora_branch.py), because the stock loader's int8 merge keeps little
        # of FlashGen's delta (bench/results/2026-09-26_int8_lora_requant.json)
        # and vllm-omni's native route applies it at run time too; the
        # release's schedule shifted at 12 on Euler, no guidance, as vllm-omni
        # runs it; the repo's attention default. T2VA only: vllm-omni refuses
        # FlashGen on any other task. Unjudged by the owner when it shipped;
        # h3_probe_t2v_flashgen_r64_4step is the merged control.
        ("h3_text_to_video_flashgen.json", "texttovideoflashgen", "t2v", LONG_T2V_PROMPT,
         dict(lora=(FLASHGEN_R64_LORA, FLASHGEN_STRENGTH), lora_branch=True,
              steps=FLASHGEN_STEPS, sampler_name=FLASHGEN_SAMPLER,
              manual_sigmas=FLASHGEN_MANUAL_SIGMAS,
              out_prefix="Video/text_to_video_flashgen"),
         "text -> video + audio at 4 steps via FlashGen at full rank, applied at the call, kitchen dense + Sol"),
        # PDD8 finished by FlashGen from sigma 0.8, promoted from
        # distill_experiments/ (owner, 2026-09-27: "then yeah switch it"). The
        # owner's review of the finisher grid: better than PDD8 alone on three
        # t2v scenes (sign text, noodle_bar), never worse; full and late-only
        # FlashGen finishes indistinguishable, so the full one, as trained. Same
        # wall time as PDD8 (6 PDD8 + 2 FlashGen evaluations; measured in the
        # render dataset, 2026-09-27). t2v only: on i2v the finish brightens
        # the frame at once and the owner kept PDD8.
        ("h3_text_to_video_pdd8_flashgen_finish.json", "texttovideopdd8flashgenfinish", "t2v", LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler", lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              manual_sigmas=STEP_SWITCH_REV["h080"][0], step_switch=True, step_switch_to="flashgen",
              step_switch_sigmas=STEP_SWITCH_REV["h080"][1],
              out_prefix="Video/text_to_video_pdd8_flashgen_finish"),
         "text -> video + audio: PDD8 to sigma 0.8, then FlashGen finishing (the owner's t2v pick), kitchen dense + Sol"),
        # FlashGen on its late blocks only (owner, 2026-09-27: make the "try
        # these" graphs). fastdude's FT1: blocks 34-49 alone keep the 4-step
        # finish with about half the haze; 0-33 alone is broken.
        ("h3_text_to_video_flashgen_late_blocks.json", "texttovideoflashgenlate", "t2v", LONG_T2V_PROMPT,
         dict(lora=(FLASHGEN_R64_LORA, FLASHGEN_STRENGTH), lora_branch=True, lora_blocks="34-49",
              steps=FLASHGEN_STEPS, sampler_name=FLASHGEN_SAMPLER,
              manual_sigmas=FLASHGEN_MANUAL_SIGMAS, distill_experiment=True,
              out_prefix="Video/text_to_video_flashgen_late_blocks"),
         "text -> video + audio at 4 steps via FlashGen on blocks 34-49 only (less haze), kitchen dense + Sol"),

        # FlashGen beyond T2VA, 2026-09-26, the owner: "even if it wasnt
        # trained with that, its worth testing". Both are the shipped t2v
        # graph's settings on another task. First frame runs on the same fl2va
        # checkpoint and file; ref2va runs on the ref2va checkpoint with the
        # file converted for its adaln basis. vllm-omni refuses FlashGen on
        # anything but T2VA, so both are outside the upstream contract.
        ("h3_probe_i2v_flashgen_4step.json", "i2v-flashgen-4step", "i2v", None,
         dict(lora=(FLASHGEN_R64_LORA, FLASHGEN_STRENGTH), lora_branch=True,
              steps=FLASHGEN_STEPS, sampler_name=FLASHGEN_SAMPLER,
              manual_sigmas=FLASHGEN_MANUAL_SIGMAS,
              out_prefix="Video/h3_probe_i2v_flashgen_4step"),
         "first frame + text -> video + audio at 4 steps via FlashGen, an untrained task"),
        ("h3_probe_r2v_flashgen_4step.json", "r2v-flashgen-4step", "r2v", _ref_prompt(images=True),
         dict(lora=(FLASHGEN_R64_REF2VA_LORA, FLASHGEN_STRENGTH), lora_branch=True,
              steps=FLASHGEN_STEPS, sampler_name=FLASHGEN_SAMPLER,
              manual_sigmas=FLASHGEN_MANUAL_SIGMAS,
              out_prefix="Video/h3_probe_r2v_flashgen_4step"),
         "image references -> video + audio at 4 steps via FlashGen on ref2va, an untrained transfer"),

        ("h3_text_to_video_pdd_4step.json", "texttovideopdd4step", "t2v", LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler",
              lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS_FAST,
              out_prefix="Video/text_to_video_pdd_4step"),
         "text -> video + audio at 4 steps via PDD, kitchen dense + Sol"),

        ("h3_text_to_video_pdd_manual_sigmas.json", "texttovideopddmanualsigmas",
         "t2v", LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler",
              lora=(PDD_FL2VA_LORA, PDD_STRENGTH),
              manual_sigmas=PDD_MANUAL_SIGMAS, steps=PDD_MANUAL_EVALS,
              out_prefix="Video/text_to_video_pdd_manual_sigmas"),
         "text -> video + audio on a tail-weighted PDD partition, kitchen dense + Sol"),

        # PDD8 on first frame alone (owner, 2026-09-27): the fl2va sidecar
        # covers it; until now only the first+last graph existed. The owner's
        # i2v pick over the FlashGen finish (2026-09-27), so promoted from
        # distill_experiments/ to the root (owner, 2026-10-01: "ok" to moving
        # it). Its `_savelat` twin stays in distill_experiments/.
        ("h3_first_frame_to_video_pdd.json", "firstframetovideopdd", "i2v", None,
         dict(pdd=True, sampler_name="euler",
              lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              out_prefix="Video/first_frame_to_video_pdd"),
         "first frame -> video + audio at 8 steps via PDD, kitchen dense + Sol"),
        # ... and PDD8 finished by FlashGen from sigma 0.8, the reverse switch's
        # better handoff on t2v (`2026-09-27_reverse_switch.md`).
        ("h3_probe_i2v_step_switch_pdd8_flashgen_h080.json", "i2v-step-switch-pdd8-flashgen-h080", "i2v", None,
         dict(pdd=True, sampler_name="euler", lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              manual_sigmas=STEP_SWITCH_REV["h080"][0], step_switch=True, step_switch_to="flashgen",
              step_switch_sigmas=STEP_SWITCH_REV["h080"][1],
              out_prefix="Video/h3_probe_i2v_step_switch_pdd8_flashgen_h080"),
         "first frame: PDD8 to 0.8, then FlashGen finishing"),

        ("h3_first_last_frame_to_video_pdd.json", "firstlastframetovideopdd", "i2v", None,
         dict(last_frame=True, **FL2V_CANVAS,
              pdd=True, sampler_name="euler",
              lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              out_prefix="Video/first_last_frame_to_video_pdd"),
         "first+last frame -> video + audio at 8 steps via PDD, kitchen dense + Sol"),

        ("h3_first_last_frame_to_video_pdd_4step.json", "firstlastframetovideopdd4step", "i2v", None,
         dict(last_frame=True, **FL2V_CANVAS,
              pdd=True, sampler_name="euler",
              lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS_FAST,
              out_prefix="Video/first_last_frame_to_video_pdd_4step"),
         "first+last frame -> video + audio at 4 steps via PDD, kitchen dense + Sol"),

        ("h3_image_ref_plus_text_to_video_pdd.json", "imagerefplustexttovideopdd", "r2v", _ref_prompt(images=True),
         dict(pdd=True, sampler_name="euler",
              lora=(PDD_REF2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              out_prefix="Video/image_ref_plus_text_to_video_pdd"),
         "image references -> video + audio at 8 steps via PDD, kitchen dense + Sol"),

        ("h3_image_ref_plus_text_to_video_pdd_4step.json", "imagerefplustexttovideopdd4step", "r2v", _ref_prompt(images=True),
         dict(pdd=True, sampler_name="euler",
              lora=(PDD_REF2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS_FAST,
              out_prefix="Video/image_ref_plus_text_to_video_pdd_4step"),
         "image references -> video + audio at 4 steps via PDD, kitchen dense + Sol"),


        ("h3_probe_ref2v_pdd.json", "r2v-pdd", "r2v", _ref_prompt(images=True),
         dict(pdd=True, dense_attn=True, sampler_name="euler", lora=(PDD_REF2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              length=243, out_prefix="Video/h3_probe_r2v_pdd"),
         "ref2va at 8 steps via Parallel Decoding Distillation"),

        # The control for the arm above. The measured gap between a fused head
        # and the checkpoint's own is 0.005 early and 0.015 at the last step
        # (docs/h3_pdd.md), so this is the arm that says whether that gap is
        # perceptible or merely real.
        ("h3_probe_ref2v_pdd_headfree.json", "r2v-pdd-headfree", "r2v",
         _ref_prompt(images=True),
         dict(pdd=True, dense_attn=True, sampler_name="euler", pdd_heads=False,
              lora=(PDD_REF2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              length=243, out_prefix="Video/h3_probe_r2v_pdd_headfree"),
         "PDD backbone only, the checkpoint's own output heads"),

        # Length sweep. The fused heads are indexed by time, not by call
        # count, so a longer clip changes the token budget and not the
        # schedule -- which is the property worth confirming rather than
        # assuming, because it is the one that would break silently.
        ("h3_probe_ref2v_pdd_345.json", "r2v-pdd-345", "r2v",
         _ref_prompt(images=True),
         dict(pdd=True, dense_attn=True, sampler_name="euler", lora=(PDD_REF2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              length=345, out_prefix="Video/h3_probe_r2v_pdd_345"),
         "PDD ref2va at the long end of the trained frame range"),

        ("h3_probe_ref2v_pdd_8s.json", "r2v-pdd-8s", "r2v",
         _ref_prompt(images=True),
         dict(pdd=True, dense_attn=True, sampler_name="euler", lora=(PDD_REF2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              length=192, out_prefix="Video/h3_probe_r2v_pdd_8s"),
         "PDD ref2va at exactly eight seconds"),

        # INVERTED TWICE with the default. 2026-08-28 the default went off
        # and this arm turned upscaling on; 2026-09-13 the default went back
        # on (vendor parity) and this arm is the one that turns it OFF.
        # Re-pointed rather than deleted, because the question is still open
        # and this is the graph that asks it; `bench/refview2_arms.json` is
        # the ablation that answers it across scenes.
        ("h3_probe_reference_upscale.json", "r2v-upscale", "r2v", _ref_prompt(images=True),
         dict(ref_upscale=False, out_prefix="Video/h3_probe_ref_upscale"),
         "same references, WITHOUT the reference pipeline's upscale"),

        ("h3_probe_square_canvas.json", "t2v-1to1", "t2v", LONG_T2V_PROMPT,
         dict(width=768, height=768,
              out_prefix="Video/h3_probe_square"),
         "the same prompt on the cheapest legal canvas"),

        # TWO graphs turn Sol-Attn ON. Both are probes; everything else ships
        # it bypassed. This one puts references in front of it, and exists
        # because the t2v probe below cannot verify what v2 of the CUDA node
        # changed.
        #
        # v2 narrowed `sink_q` to the target-audio rows, leaving reference
        # queries sparse. The narrowing is `audio_start // 64` blocks, and on
        # t2v `audio_start` IS the text length -- measured 311 rows on the
        # shipped graph, so 4 blocks. Four is a real signal and too thin to
        # trust: an off-by-one in the block arithmetic would be
        # indistinguishable from success, and v2's `audio is None` fallback
        # silently reproduces v1's `(0, N)`. With references the sink is
        # thousands of rows, so the narrowing is tens of blocks and unmissable.
        #
        # Paired with `h3_text_to_video.json` deliberately: same canvas, same
        # length, same seed, same Sol settings, references the only variable.
        # Read the `conditioning sink` line from both.
        #
        # This is a MECHANISM probe, not a speed one, and the distinction is
        # load-bearing after 2026-08-14. Reference rows are pinned exact, so
        # they raise the token count without adding anything Sol can sparsify
        # -- arithmetic over the measured row counts puts a video-reference
        # arm's attention ceiling near 1.58x against t2v's ~8x. Reference-heavy
        # work is where Sol has the LEAST room, not the most, which is the
        # opposite of what this repo assumed for weeks. Do not read a slow
        # result here as Sol underperforming.
        ("h3_probe_sol_on_refs.json", "r2v-sol", "r2v", _ref_prompt(images=True),
         dict(sol_on=True, out_prefix="Video/h3_probe_sol_on_refs"),
         "reference images with Sol-Attn ON -- the sink at reference load"),

        # `h3_probe_sol_on` and `h3_probe_sol_on_i2v` stood here until
        # 2026-09-14 (owner). Sol went on by default after they were written,
        # so each matched `h3_text_to_video` or `h3_first_frame_to_video` in
        # everything but its output name; those two are the text-only and
        # keyframe Sol arms now.

        # Sol WITHOUT sage. The owner, 2026-09-04: "maybe it's better to try
        # without sage at all". Every other Sol graph chains Sol over sage, so
        # the steps outside Sol's window and every call Sol declines run sage;
        # here they run ComfyUI's stock attention. Two uses: a blind arm, and,
        # on an armed server, a probe record whose counterfactual is stock
        # attention rather than sage (bench/check_sol_probe.py), which is the
        # first direct Sol-against-near-exact measurement the repo would hold.
        ("h3_probe_t2v_sol_nosage.json", "t2v-sol-nosage", "t2v", LONG_T2V_PROMPT,
         dict(dense_attn="sol", out_prefix="Video/h3_probe_t2v_sol_nosage"),
         "text -> video + audio, Sol as shipped, no sage: stock attention outside Sol"),

        # `h3_probe_t2v_sol_core` stood here from 2026-09-10 to 2026-09-15:
        # core's `BlockSparseAttention` at its own defaults over sage, driven
        # by the core-versus-ours A/B. Retired with that manifest once the
        # default floor became the kitchen backend, because a re-run would have
        # set ours on kitchen against core's node on sage under the A/B's name.
        # Its clips were rendered and never scored; the record lists them
        # (bench/results/2026-09-10_sol_core_ab_outputs.json). `sol_impl="core"`
        # stays for the next such arm, core's node over the backend node.

        # **Block-49 probes, 2026-09-15** (docs/h3_block49_quant_error.md).
        # Same prompt and seed as the shipped t2v graph, so they form
        # same-seed pairs against it. `exact_tail` runs blocks 45, 48 and 49
        # on ComfyUI's own bf16 attention: the ceiling of what any fix at
        # those blocks can do, at a few percent of render time. It, `levers`
        # and `policy` were rendered and scored on the sage chain, the default
        # that morning, so they stay on it ("sage_sol") now that the default
        # is the kitchen chain, and `exact_tail` keeps Sol's balance off as it
        # rendered. `balanced` (the balance node plus sage's qk_balance) stood
        # here until the flip: scored, and superseded by `levers`.
        ("h3_probe_t2v_exact_tail.json", "t2v-exact-tail", "t2v", LONG_T2V_PROMPT,
         dict(dense_attn="sage_sol", sol_overrides={"quantizer": "plain", "dense_blocks": ""},
              exact_blocks="45,48,49", out_prefix="Video/h3_probe_t2v_exact_tail"),
         "text -> video + audio, the sage chain with blocks 45/48/49 on exact bf16 attention"),
        # docs/h3_quant_policy.md. `levers` is Tier 1's witness: every free
        # lever on (the balance node, sage's qk_balance, Sol's qk_balance)
        # and NO exact blocks, the graph that has to match `exact_tail` for
        # the bf16 row to disappear. `policy` is Tier 0/1: the same levers
        # plus the three lopsided blocks on exact attention. Sol's qk_balance
        # is the recipe's own since the flip, so neither overrides it.
        # The community chain (2026-09-15): kitchen's rotated INT8 kernel on
        # the dense steps, Sol on the routed ones, no sage. The default since
        # that evening, so `ck` is now the arm that departs from it: Sol's
        # qk_balance off and no dense tail, the chain as most people run it.
        # `ck_balanced` stood here until the flip made it the default.
        # `h3_probe_t2v_ck_dense_tail` (blocks 45/48/49 on the dense kernel)
        # stood here until 2026-09-25, when the owner made that tail the
        # recipe's `dense_blocks`; its pair survives inverted as
        # `h3_probe_t2v_no_dense_tail`, the default with the tail back on Sol.
        ("h3_probe_t2v_ck.json", "t2v-ck", "t2v", LONG_T2V_PROMPT,
         dict(dense_attn="ck", sol_overrides={"quantizer": "plain", "dense_blocks": ""},
              out_prefix="Video/h3_probe_t2v_ck"),
         "text -> video + audio, community chain as most run it: kitchen int8 attention dense + Sol, qk_balance off, no dense tail"),
        ("h3_probe_t2v_no_dense_tail.json", "t2v-no-dense-tail", "t2v", LONG_T2V_PROMPT,
         dict(dense_attn="ck", sol_overrides={"dense_blocks": ""},
              out_prefix="Video/h3_probe_t2v_no_dense_tail"),
         "text -> video + audio, the default chain with blocks 45/48/49 back on Sol (the pre-2026-09-25 default)"),
        # Tier 2 witness (2026-09-15 night): the default chain with Sol's
        # Hadamard rotation on (docs/h3_quant_policy.md). Same seed as every
        # other market arm today, so it sits beside h3_text_to_video (the
        # default, rotate off) and h3_probe_t2v_no_dense_tail.
        # Fully dense control (2026-09-15 night): neither Sol nor any INT8
        # kernel, ComfyUI's own attention on every step. The ceiling for the
        # whole chain and the arm that says whether a flaw every INT8 arm
        # shares (the market coins appearing from nowhere on all three
        # community-chain arms) belongs to the kernels or to the take.
        ("h3_probe_t2v_dense.json", "t2v-dense", "t2v", LONG_T2V_PROMPT,
         dict(dense_attn=True, out_prefix="Video/h3_probe_t2v_dense"),
         "text -> video + audio, fully dense: no sage, no Sol, no INT8 anywhere"),
        ("h3_probe_t2v_rotate.json", "t2v-rotate", "t2v", LONG_T2V_PROMPT,
         dict(dense_attn="ck", sol_overrides={"quantizer": "balanced+rotated"}, out_prefix="Video/h3_probe_t2v_rotate"),
         "text -> video + audio, the default chain with Sol rotate on"),
        # RECORDS, not recommendations (2026-09-18). The three graphs below
        # stack MiniMaxH3ChannelBalance under sage's own `fp8++ balanced`.
        # Graded 2026-09-17, the fold adds nothing there (sage rebalances per
        # head inside its quantizer) and it re-rounds q/k in bf16, which moves
        # exact attention on the last block a little
        # (bench/results/2026-09-17_channel_balance_vs_sage_balanced_b49_s15.json).
        # They stay byte for byte because dated records and clips were rendered
        # from them and renders repeat bit for bit: `levers` is the sage arm of
        # bench/results/2026-09-15_block49_repro_batch.md, `sage_rotate` was
        # proven identical across two days on 2026-09-17. For a sage chain to
        # USE, build the sage set (`--chain sage`, h3_config.DENSE_CHAINS),
        # which has no fold.
        ("h3_probe_t2v_levers.json", "t2v-levers", "t2v", LONG_T2V_PROMPT,
         dict(dense_attn="sage_sol", channel_balance="loud blocks (from weights)",
              sage_mode="fp8++ balanced", out_prefix="Video/h3_probe_t2v_levers"),
         "text -> video + audio, the sage chain with every free lever: balance node + sage and Sol qk_balance, no exact blocks"),
        # The sage chain with every lever AND Sol's rotation (2026-09-15 night,
        # owner's ask): the best the sage chain can do against the kitchen
        # chain, same seed as every other market arm.
        ("h3_probe_t2v_sage_rotate.json", "t2v-sage-rotate", "t2v", LONG_T2V_PROMPT,
         dict(dense_attn="sage_sol", channel_balance="loud blocks (from weights)",
              sage_mode="fp8++ balanced", sol_overrides={"quantizer": "balanced+rotated"},
              out_prefix="Video/h3_probe_t2v_sage_rotate"),
         "text -> video + audio, the sage chain with every free lever plus Sol rotate, no exact blocks"),
        ("h3_probe_t2v_policy.json", "t2v-policy", "t2v", LONG_T2V_PROMPT,
         dict(dense_attn="sage_sol", channel_balance="loud blocks (from weights)",
              sage_mode="fp8++ balanced",
              exact_blocks="45,48,49", out_prefix="Video/h3_probe_t2v_policy"),
         "text -> video + audio, the sage-chain block-49 policy: balance node + sage and Sol qk_balance + blocks 45/48/49 exact"),

        # **Candidates on trial, 2026-09-05.** The owner asked for canonical
        # graphs carrying the settings the lane currently thinks are its
        # leaders, to render their own prompts on. The evidence stands as
        # docs/roadmap.md (forward plan 2026-09-04) records it: the shipped
        # h3_text_to_video (sage + Sol) is the leader with a verdict behind
        # it; each candidate below is a configuration blinded but not yet
        # judged, and its note says what it costs and what would settle it.
        # PDD8 under sage alone is the fourth candidate and already exists as
        # h3_probe_t2v_pdd8_sage. Every overridden Sol field is declared in
        # bench/check_attention_defaults.py::DEVIATIONS.
        ("h3_candidate_t2v_sol_only.json", "t2v-candidate-sol-only", "t2v", LONG_T2V_PROMPT,
         dict(dense_attn="sol",
              out_prefix="Video/h3_candidate_t2v_sol_only"),
         "CANDIDATE text -> video + audio: Sol only, from the recipe's start_percent (its 0.0 override removed 2026-10-05, owner: no 0.0 Sol start anywhere)"),

        ("h3_candidate_t2v_pdd8_sol_narrow.json", "t2v-candidate-pdd8-sol-narrow", "t2v", LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler",
              lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              sol_overrides={"start_percent": 0.3, "end_percent": 0.6},
              out_prefix="Video/h3_candidate_t2v_pdd8_sol_narrow"),
         "CANDIDATE text -> video + audio at 8 steps via PDD, Sol on two steps"),

        # The audio-freeze lane on the fast chain (owner, 2026-09-12: "worth
        # testing if PDD works with this since it's faster iteration", which
        # reopens the parked PDD lane for this use only). Same graph as the
        # candidate below plus the freeze node; the documented PDD weakness is
        # its audio (docs/research/pdd/audio_under_pdd.md), and a frozen track
        # takes the audio rows out of what PDD has to get right.
        ("h3_candidate_t2v_pdd8_baked_audio_freeze.json", "t2v-candidate-pdd8-baked-audio-freeze",
         "t2v", LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler",
              unet=MODELS["unet_fl2va_pdd8_baked"],
              lora=(PDD_FL2VA_STRIPPED_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              freeze_audio=True,
              out_prefix="Video/h3_candidate_t2v_pdd8_baked_audio_freeze"),
         "CANDIDATE text + a frozen audio track -> video at 8 steps via PDD"),

        # The same on the PDD8 chain with the track also anchored as guide
        # rows (owner, 2026-09-12: the untold arm works but is less natural;
        # music video wants no transcript, so give the model more of the
        # track without words). API only.
        ("h3_candidate_t2v_pdd8_baked_audio_freeze_guide.json", "t2v-candidate-pdd8-baked-audio-freeze-guide",
         "t2v", LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler",
              unet=MODELS["unet_fl2va_pdd8_baked"],
              lora=(PDD_FL2VA_STRIPPED_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              freeze_audio=True, freeze_guide=True, out_prefix="Video/h3_candidate_t2v_pdd8_baked_audio_freeze_guide"),
         "CANDIDATE text + a frozen audio track, also anchored as guide rows -> video at 8 steps via PDD"),

        ("h3_candidate_t2v_pdd8_baked.json", "t2v-candidate-pdd8-baked", "t2v", LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler",
              unet=MODELS["unet_fl2va_pdd8_baked"],
              lora=(PDD_FL2VA_STRIPPED_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              out_prefix="Video/h3_candidate_t2v_pdd8_baked"),
         "CANDIDATE text -> video + audio at 8 steps via PDD on the baked checkpoint, attention as shipped"),

        # Sol-Attn ON at full reference load: images + a reference video + its
        # soundtrack + standalone audio. This is the heaviest sink the model
        # accepts, and it is the workload the owner actually renders -- the
        # tau/morton/centroid_tail arms moved here from t2v on 2026-08-14 for
        # exactly that reason.
        #
        # It matters for Sol specifically because every reference row is pinned
        # exact as a KEY at any tau, so this is where the sink is largest and
        # where v2's narrowing has the most to do. It is also where Sol has the
        # LEAST headroom: pinned rows raise the token count without adding
        # anything sparsifiable, so read it as a mechanism and quality arm, not
        # a speed one.
        ("h3_probe_sol_on_all_refs.json", "r2v-all-sol", "r2v",
         _ref_prompt(images=True, video=True, video_audio=True, audio=True),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_audio=True, sol_on=True,
              out_prefix="Video/h3_probe_sol_on_all_refs"),
         "every reference type at once, with Sol-Attn ON"),

        # The step-caching arm. Same references, same Sol config, same budget
        # as h3_probe_sol_on_all_refs -- that graph IS the control; this one
        # adds only the EasyCache node, so a timing pair between the two
        # varies exactly one thing. Threshold sweeps patch the widget at
        # submit time rather than multiplying graphs.
        ("h3_probe_cache_easy.json", "r2v-all-cache", "r2v",
         _ref_prompt(images=True, video=True, video_audio=True, audio=True),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_audio=True, sol_on=True,
              cache=CACHE_NODE,
              out_prefix="Video/h3_probe_cache_easy"),
         "the all-refs Sol arm plus EasyCache step reuse"),

        # Until 2026-10-05 an "euler pair" followed here (`h3_probe_euler`,
        # `h3_probe_euler_cache`, owner-requested 2026-08-18): the two graphs
        # above with only the sampler changed, because the base then sampled
        # on `er_sde`, which re-noises and reused nothing at the stock cache
        # threshold (bench/results/2026-08-18_cache_arms.jsonl). The base
        # runs Euler now (h3_config.SAMPLING), so the pair became those two
        # graphs byte for byte and was retired: `h3_probe_sol_on_all_refs`
        # against `h3_probe_cache_easy` is the deterministic cache pair.

        # --- the single-frame image gen/edit path -------------------------
        #
        # Every graph below renders ONE FRAME and is written to
        # `workflows/image/` rather than beside the video graphs -- the split
        # is by use case, and `_graph_dir` derives it from `single_frame` so
        # there is no second place to keep in sync. Video is the primary case;
        # this one is experimental and moves faster.
        #
        # They come last for the reason they always did: appending is the habit
        # that keeps saved graphs working.
        #
        # `ref_images` names the scene's own references from the documented
        # `h3_refs/` library instead of the two root placeholders, so a result
        # is attributable to a subject somebody can look up in
        # `internal/reference_library.md`.
        # **The single-frame image graphs are parked, 2026-08-27.** Not emitted,
        # not discovered, not graded; the last generated set is
        # `archive/workflows/image/` and the shim they need is
        # `archive/single_frame.py`. Their builder, `_image_graphs()`, left this
        # file with the UI half on 2026-09-14, so restoring the lane starts
        # from the archive and git history. See `docs/h3_image_editing.md`.

        ("h3_probe_head_chunks.json", "t2v-chunk4", "t2v", LONG_T2V_PROMPT,
         dict(head_chunks=4, dense_attn="sage_sol", out_prefix="Video/h3_probe_chunk4"),
         "the same render with the heads in 4 groups"),
    )

    # The 2026-09-26 distill run (owner): `_savelat` probe twins of exactly the
    # graphs the run renders, each saving the latent its decoders read, so
    # later routing experiments reuse these latents without re-rendering. Built
    # from the source entry, so a twin cannot drift from it. The two refine
    # twins also carry the frozen-row test, and the route 3 step-switch arm
    # comes with its own twin.
    _SAVELAT_OF = ("h3_text_to_video", "h3_text_to_video_pdd", "h3_text_to_video_flashgen",
                   "h3_probe_t2v_fasth3_8step_contract",
                   "h3_probe_t2v_flashgen_r64_4step_branch_dense",
                   "h3_probe_t2v_pdd8_audio_refine", "h3_probe_t2v_flashgen_4step_audio_refine")
    _by_name = {e[0]: e for e in GRAPHS}
    _step_switch = (
        "h3_probe_t2v_step_switch_flashgen_pdd8.json", "t2v-step-switch-flashgen-pdd8", "t2v",
        LONG_T2V_PROMPT,
        dict(lora=(FLASHGEN_R64_LORA, FLASHGEN_STRENGTH), lora_branch=True,
             steps=FLASHGEN_STEPS, sampler_name=FLASHGEN_SAMPLER,
             manual_sigmas=STEP_SWITCH_PASS1_SIGMAS, step_switch=True,
             out_prefix="Video/h3_probe_t2v_step_switch_flashgen_pdd8"),
        "route 3: FlashGen's first two steps, then PDD8's finish from 0.888889 (h3_config.STEP_SWITCH_*)")
    # The PDD schedule tests (docs/h3_distills.md, "Tests that would move this
    # section", 2026-09-26): PDD at widths 2 and 1 beside the shipped 8, 6 and
    # 4, and the base on Euler over PDD's own 32-point grid, the path its heads
    # were distilled from. PDD at width 1 is the student's own per-interval
    # path, so the two 32-step graphs isolate fusion error from following.
    _pdd_tests = tuple(
        (f"h3_probe_t2v_pdd{n}.json", f"t2v-pdd{n}", "t2v", LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler", lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=n,
              out_prefix=f"Video/h3_probe_t2v_pdd{n}"),
         f"PDD at {n} steps (width {32 // n}) on its own grid, for the schedule tests")
        for n in (16, 32)) + (
        ("h3_probe_t2v_base_euler32.json", "t2v-base-euler32", "t2v", LONG_T2V_PROMPT,
         dict(sampler_name="euler", steps=32, out_prefix="Video/h3_probe_t2v_base_euler32"),
         "the base on Euler at 32 steps: PDD's teacher path, sharing the distills' starting noise"),
        # The fair base-versus-distill pair (P4 in
        # bench/results/2026-09-26_distill_run_predictions.md) was
        # `h3_probe_t2v_base_euler16`: the base at its own 16 steps, but on
        # Euler, so it shared the distills' starting noise. Since 2026-10-05
        # that is `h3_text_to_video` itself (h3_config.SAMPLING), so the probe
        # and its `_savelat` twin were retired as duplicates of it.
        # FastH3 at its contract with VSA off (dense on the kitchen backend), for
        # the "low bitrate" texture question (docs/h3_distills.md): off its
        # training regime by design, a probe only.
        ("h3_probe_t2v_fasth3_8step_contract_novsa.json", "t2v-fasth3-8step-contract-novsa", "t2v",
         LONG_T2V_PROMPT,
         dict(dense_attn="ck", sol_on=False, unet=MODELS["unet_fasth3_v2"],
              steps=FASTH3_STEPS, sampler_name=FASTH3_CONTRACT_SAMPLER,
              manual_sigmas=FASTH3_CONTRACT_SIGMAS, shift=FASTH3_SHIFT,
              out_prefix="Video/h3_probe_t2v_fasth3_8step_contract_novsa"),
         "FastH3 V2 on its contract's sampling with VSA off (dense): the texture probe"),)
    # The PDD8 finisher grid (2026-09-27, the owner's consolidation) renders on
    # _savelat twins so any clip can be re-encoded for review without a render
    # (#38: the 8-bit save adds dark blocking mainly to PDD8).
    _savelat_more = ("h3_text_to_video_pdd_4step", "h3_text_to_video_pdd_manual_sigmas",
                     "h3_text_to_video_flashgen_late_blocks", "h3_first_frame_to_video_pdd",
                     "h3_probe_i2v_step_switch_pdd8_flashgen_h080")
    _twins = []
    for fname, label, task, prompt, extra, note in (
            [_by_name[f + ".json"] for f in _SAVELAT_OF + _savelat_more] + [_step_switch] + list(_pdd_tests)):
        stem = fname.removesuffix(".json")
        twin_extra = dict(extra, save_latents=True,
                          out_prefix=extra.get("out_prefix", f"Video/{stem}") + "_savelat")
        if extra.get("audio_refine"):
            twin_extra["probe_frozen_rows"] = True
        _twins.append((f"{stem}_savelat.json", f"{label}-savelat", task, prompt, twin_extra,
                       f"{note}; saves its latents (the 2026-09-26 distill run)"))
    for fname, label, task, prompt, extra, note in [_by_name["h3_text_to_video_pdd.json"], _pdd_tests[2]]:  # PDD8, base-Euler-32
        stem = fname.removesuffix(".json")
        _twins.append((f"{stem}_x0.json", f"{label}-x0", task, prompt,
                       dict(extra, save_latents=True, probe_step_x0=True,
                            out_prefix=extra.get("out_prefix", f"Video/{stem}") + "_x0"),
                       f"{note}; saves every step's x0 and the final latent (one or two renders only)"))
    # FastH3's contract graph with a capture node on core's sparse path
    # (open_experiments #45, 2026-09-27): inert unless H3_CAPTURE is set with
    # pre=, and saves its latents so the captured render can be checked.
    _f = _by_name["h3_probe_t2v_fasth3_8step_contract.json"]
    _twins.append(("h3_probe_t2v_fasth3_8step_contract_capture.json", _f[1] + "-capture", _f[2], _f[3],
                   dict(_f[4], save_latents=True, probe_core_sparse_capture=True,
                        out_prefix=_f[4].get("out_prefix", "Video/h3_probe_t2v_fasth3_8step_contract") + "_capture"),
                   f"{_f[5]}; captures q/k/v on core's sparse path when H3_CAPTURE is armed (#45)"))
    # The reverse step switch, one per handoff (h3_config.STEP_SWITCH_REV),
    # each with its savelat twin. PDD first, FlashGen finishing.
    _reverse = tuple(
        (f"h3_probe_t2v_step_switch_pdd8_flashgen_{h}.json", f"t2v-step-switch-pdd8-flashgen-{h}", "t2v",
         LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler", lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              manual_sigmas=STEP_SWITCH_REV[h][0], step_switch=True, step_switch_to="flashgen",
              step_switch_sigmas=STEP_SWITCH_REV[h][1],
              out_prefix=f"Video/h3_probe_t2v_step_switch_pdd8_flashgen_{h}"),
         f"reverse step switch: PDD8 to {STEP_SWITCH_REV[h][1].split(',')[0]}, then FlashGen finishing")
        for h in STEP_SWITCH_REV)
    # The same switch with FlashGen on its late blocks only (open_experiments
    # #34, 2026-09-27): FT1 found blocks 34-49 carry the 4-step finish and the
    # early blocks the haze, so this finisher should lift PDD8's highlights
    # without adding haze. h080 only; its full-FlashGen twin is the control.
    _reverse = _reverse + tuple(
        (f"h3_probe_t2v_step_switch_pdd8_flashgen_late_{h}.json", f"t2v-step-switch-pdd8-flashgen-late-{h}", "t2v",
         LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler", lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              manual_sigmas=STEP_SWITCH_REV[h][0], step_switch=True, step_switch_to="flashgen",
              step_switch_sigmas=STEP_SWITCH_REV[h][1], step_switch_blocks="34-49",
              out_prefix=f"Video/h3_probe_t2v_step_switch_pdd8_flashgen_late_{h}"),
         f"reverse step switch: PDD8 to {STEP_SWITCH_REV[h][1].split(',')[0]}, then FlashGen on blocks 34-49 finishing")
        for h in ("h080",))
    # PDD8 finished by FastH3's own checkpoint (2026-09-29, the owner via
    # mutant): a second full checkpoint between the passes, and a shift that
    # decides the audio (h3_config.STEP_SWITCH_FASTH3). Same handoff as h080.
    _reverse = _reverse + tuple(
        (f"h3_probe_t2v_step_switch_pdd8_fasth3_{k}.json", f"t2v-step-switch-pdd8-fasth3-{k}", "t2v",
         LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler", lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              manual_sigmas=STEP_SWITCH_FASTH3[k][0], step_switch=True, step_switch_to="fasth3",
              step_switch_sigmas=STEP_SWITCH_FASTH3[k][1], step_switch_shift=STEP_SWITCH_FASTH3[k][2],
              out_prefix=f"Video/h3_probe_t2v_step_switch_pdd8_fasth3_{k}"),
         f"step switch: PDD8 to 0.8, then FastH3's checkpoint at {STEP_SWITCH_FASTH3[k][1].split(',')[0]} "
         f"(shift {STEP_SWITCH_FASTH3[k][2]['shift_video']:g}/3) finishing")
        for k in STEP_SWITCH_FASTH3)
    # The same two finishes on ref2va (2026-09-29, the owner): PDD8 on ref2va
    # with one reference image, then FlashGen's ref2va LoRA (the control) or
    # FastH3's own checkpoint, which is fl2va-derived and has never seen a
    # reference token: an untrained transfer, like FlashGen on ref2va.
    _r2v_switch = dict(ref_images=MARKET_REF_IMAGES, ref_image_count=1, pdd=True, sampler_name="euler",
                       lora=(PDD_REF2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
                       manual_sigmas=STEP_SWITCH_REV["h080"][0], step_switch=True)
    _reverse = _reverse + (
        ("h3_probe_r2v_step_switch_pdd8_flashgen_h080.json", "r2v-step-switch-pdd8-flashgen-h080", "r2v",
         MARKET_REF2V_PROMPT,
         dict(_r2v_switch, step_switch_to="flashgen", step_switch_sigmas=STEP_SWITCH_REV["h080"][1],
              out_prefix="Video/h3_probe_r2v_step_switch_pdd8_flashgen_h080"),
         "market scene as ref2va: PDD8 to 0.8, then FlashGen finishing"),
    ) + tuple(
        (f"h3_probe_r2v_step_switch_pdd8_fasth3_{k}.json", f"r2v-step-switch-pdd8-fasth3-{k}", "r2v",
         MARKET_REF2V_PROMPT,
         dict(_r2v_switch, step_switch_to="fasth3", step_switch_sigmas=STEP_SWITCH_FASTH3[k][1],
              step_switch_shift=STEP_SWITCH_FASTH3[k][2],
              out_prefix=f"Video/h3_probe_r2v_step_switch_pdd8_fasth3_{k}"),
         f"market scene as ref2va: PDD8 to 0.8, then FastH3's checkpoint at {STEP_SWITCH_FASTH3[k][1].split(',')[0]} "
         f"(shift {STEP_SWITCH_FASTH3[k][2]['shift_video']:g}/3) finishing")
        for k in STEP_SWITCH_FASTH3)
    # PDD8 finished by the undistilled base (open_experiments #37, 2026-09-27):
    # whether PDD8's dim highlights are its coarse tail's, with no second
    # distill in the finish. Same handoff as rev_h063.
    _reverse = _reverse + tuple(
        (f"h3_probe_t2v_step_switch_pdd8_base_{h}.json", f"t2v-step-switch-pdd8-base-{h}", "t2v",
         LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler", lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              manual_sigmas=STEP_SWITCH_BASE[h][0], step_switch=True, step_switch_to="base",
              step_switch_sigmas=STEP_SWITCH_BASE[h][1],
              out_prefix=f"Video/h3_probe_t2v_step_switch_pdd8_base_{h}"),
         f"step switch: PDD8 to {STEP_SWITCH_BASE[h][1].split(',')[0]}, then the base on Euler finishing")
        for h in STEP_SWITCH_BASE)
    for fname, label, task, prompt, extra, note in _reverse:
        stem = fname.removesuffix(".json")
        _twins.append((f"{stem}_savelat.json", f"{label}-savelat", task, prompt,
                       dict(extra, save_latents=True, out_prefix=extra["out_prefix"] + "_savelat"),
                       f"{note}; saves its latents"))
    # What the owner renders with (2026-10-03): PDD8 to 0.8, then FlashGen
    # finishing, on each task. Derived from the probe entry, so a daily graph
    # cannot drift from the probe it was promoted from; made after the twins,
    # so it has none. The probes and their `_savelat` twins keep their names
    # and folder: arms files, the reproduction guard and dated records cite
    # them.
    _probes = {e[0]: e for e in GRAPHS + _reverse}
    _daily = tuple(
        (f"{name}.json", name.removeprefix("h3_").replace("_", "-"), _probes[src][2], _probes[src][3],
         dict(_probes[src][4], daily=True, out_prefix=f"Video/{name}"),
         f"daily: {_probes[src][5]}")
        for name, src in DAILY_GRAPHS.items())
    GRAPHS = GRAPHS + (_step_switch,) + _pdd_tests + _reverse + tuple(_twins) + _daily

    if args.list_scenes:
        for name, text in T2V_SCENES.items():
            first = next(l for l in text.splitlines() if l and not l.endswith(":"))
            print(f"{name:<12} {first[:64]}")
        return 0

    if args.print_scene:
        if args.print_scene not in T2V_SCENES:
            raise SystemExit(f"no scene named {args.print_scene!r}. "
                             f"Have: {', '.join(T2V_SCENES)}")
        # JSON, not raw: `--set` parses its VALUE as JSON, and these prompts
        # carry newlines and quotes that a raw paste would break on.
        print(json.dumps(T2V_SCENES[args.print_scene]))
        return 0

    if args.dump_prompts:
        # The authoritative `api filename -> prompt` map.
        #
        # Exists so a consumer can compare a shipped graph against ITS OWN
        # expected prompt. `bench/check_ref_prompt_labels.py` used to
        # re-enumerate `_ref_prompt`'s whole argument space and assert set
        # membership, which failed two ways: the enumeration had to be widened
        # by hand every time the signature grew (missed `images` becoming a
        # tuple in 2026-08-16, missed `scene=` until 2026-08-28), and set
        # membership cannot tell two arms apart -- an arm carrying a DIFFERENT
        # arm's prompt was a legal string and passed green. Both classes are
        # gone if the comparison is per graph, and this is the only place that
        # knows which prompt belongs to which graph.
        # `length`/`last_frame`/`first_frame` come from the BUILDER'S OWN
        # signature, not from constants repeated here, so a changed default
        # moves this with it instead of leaving it quietly wrong.
        import inspect as _inspect
        _p = _inspect.signature(build_api).parameters
        print(json.dumps({
            fname.removesuffix(".json") + "_api.json":
                resolve_default_prompt(
                    task, prompt,
                    length=graph_length(extra),
                    last_frame=extra.get("last_frame", _p["last_frame"].default),
                    first_frame=extra.get("first_frame",
                                          _p["first_frame"].default))
            for fname, _label, task, prompt, extra, _note in GRAPHS}))
        return 0

    if args.dump_graphs:
        # The authoritative `api path -> what the graph is` map, for
        # `bench/build_index.py`'s workflows index. `GRAPHS` is local to this
        # function and grows after its literal (`_daily`, `_twins`, ...), so
        # no reader outside can see the note any other way. Paths come from
        # `_graph_dir`, the same call the writer makes, so a graph that moves
        # folder moves here with it.
        print(json.dumps({
            str((_graph_dir(out, extra, fname) / fname.replace(".json", "_api.json"))
                .relative_to(out)):
                {"label": label, "task": task, "note": note}
            for fname, label, task, _prompt, extra, note in GRAPHS}))
        return 0

    if args.list_prompts or args.print_prompt:
        want = (args.print_prompt or "").removesuffix(".json").removeprefix("h3_")
        hit = False
        for fname, label, task, prompt, _extra, note in GRAPHS:
            short = fname.removesuffix(".json").removeprefix("h3_")
            text = prompt if prompt is not None else {
                "t2v": T2V_PROMPT, "i2v": I2V_PROMPT, "r2v": R2V_PROMPT}[task]
            if args.list_prompts:
                print(f"{short:<34} {label:<20} {text.splitlines()[0][:44]}")
            elif short == want or label == want:
                print(text)
                hit = True
        if args.print_prompt and not hit:
            raise SystemExit(
                f"no graph named {args.print_prompt!r}. "
                f"Run --list-prompts to see them.")
        return 0

    # SOL IS ALWAYS ON BY DEFAULT in every video workflow (except image workflows).
    # The owner's standing direction (2026-08-17): Sol-Attn is always default set to ON
    # on every workflow we use (except pure image single-frame workflows).
    # If a specific test needs to bypass it, it can be bypassed explicitly with sol_on=False,
    # but the canonical shipped default across all video workflows is ON.
    # Every shipped graph is API format: the editor loads one by input name
    # and arranges it, and every runner drives it over /prompt.
    alt_chain = args.chain != DEFAULT_DENSE_CHAIN
    if alt_chain and Path(args.out).resolve() == HERE.resolve():
        raise SystemExit(
            f"--chain {args.chain} would write over the shipped tree, which every check "
            f"reads as the {DEFAULT_DENSE_CHAIN!r} chain. Give it its own --out, or change "
            "h3_config.DEFAULT_DENSE_CHAIN to move the shipped tree itself.")
    for fname, label, task, prompt, extra, note in GRAPHS:
        if alt_chain and not _takes_default_chain(extra):
            continue
        # Applied on the shipped chain too, so that changing
        # h3_config.DEFAULT_DENSE_CHAIN moves every key of the chain (the
        # sage chain's mode, not only its node) rather than the node alone.
        if _takes_default_chain(extra):
            extra = _on_chain(extra, args.chain)
        sage_on, sol_on, _dense_mode = _attention_plan(extra)
        api_extra = {k: v for k, v in extra.items()
                     if k not in ("sol_on", "dense_attn", "sol_overrides", "distill_experiment", "daily")}
        # The instrumented twins keep one shared copy of a still: their
        # renders are compared byte for byte with earlier ones
        # (h3_config.REF_QWEN_SHORT_EDGE).
        if fname.removesuffix(".json").endswith(("_savelat", "_x0")):
            api_extra.setdefault("ref_qwen_short_edge", 0)
        wf = build_api(task, sage=sage_on,
                       prompt=prompt,
                       sol=(_sol_with_overrides(extra) if sol_on else None),
                       **({"dense_backend": DENSE_BACKEND_NODE["attention"]}
                          if _dense_mode == "ck" else {}),
                       **{**api_extra, "length": graph_length(api_extra)})
        if alt_chain:
            _suffix_output_prefixes(wf, args.chain)
        p = _graph_dir(out, extra, fname) / fname.replace(".json", "_api.json")
        written.append((label, p, wf))
        print(f"  {p.name}: {note}")

    # Decodes a saved latent pair (`draft_decode` or `save_latents`) with no
    # sampling. Its latent names are placeholders in the shape SaveLatent writes
    # for a `_savelat` twin's first run; set both to the pair a run wrote (the
    # counters advance together). The shipped draft graph was retired on
    # 2026-09-26 (owner: not worth it); the switch and this graph stay.
    if not alt_chain:
        _saved = "latents/text_to_video_flashgen_savelat"
        wf = build_decode_saved_latent(f"{_saved}_video_00001_.latent [output]",
                                       f"{_saved}_audio_00001_.latent [output]")
        p = out / "distill_experiments" / "h3_decode_saved_latent_api.json"
        written.append(("decodesavedlatent", p, wf))
        print(f"  {p.name}: a draft's saved latents through the real video decoder, no sampling")

    # Bench copies carrying MiniMaxH3ProvenanceStamp. Deliberately NOT the
    # shipped graphs: the stamp reads another pack's closure internals, so it
    # breaks when that pack changes, and a bench is where breakage is cheap.
    bench = out / "bench"
    if not alt_chain:
        bench.mkdir(parents=True, exist_ok=True)
    # The t2v bench pair renders BENCH_T2V_PROMPT, a bank scene chosen for
    # its failure surfaces (a figure at distance, a painted sign, dialogue,
    # a silent bystander) and NOT the covered-market scene the shipped t2v
    # graphs carry: until 2026-09-03 every bench and capture number came
    # from that one scene. Owner's call; the pair shares prompt, seed and
    # length so the two are a matched pair.
    #
    # `_dense_stamped` is the TRUE BASELINE: no sage node, no Sol node, no
    # LoRA -- the DiT and encoder every graph loads under ComfyUI's stock
    # attention at the base step count, i.e. the render you would otherwise
    # make on this box. Until 2026-09-17 a SageChainAssert in `require_absent`
    # mode made this render raise if anything patched attention on the model;
    # no generated graph carries that node now, so the baseline is a baseline
    # by its wiring alone. The `_stamped` graph beside it is the repo's older "dense"
    # convention, sage alone; every speedup number before 2026-09-03 was
    # relative to that, not to this. Outside check_attention_defaults'
    # scope like every bench graph.
    # The bench copies name their own kernels and belong to the shipped tree
    # alone; an alternate-chain set is the default-taking graphs and nothing else.
    for fname, task, prompt, sage in () if alt_chain else (
        ("h3_text_to_video_dense_stamped_api.json", "t2v", BENCH_T2V_PROMPT, False),
        ("h3_text_to_video_stamped_api.json", "t2v", BENCH_T2V_PROMPT, True),
        ("h3_image_ref_plus_text_to_video_stamped_api.json", "r2v", None, True),
        ("h3_first_frame_to_video_stamped_api.json", "i2v", None, True),
    ):
        wf = build_api(task, sage=sage, length=LONG_LENGTH,
                       sol=None, prompt=prompt, stamp=True,
                       # A bench graph is an instrument: one shared copy, as
                       # every record taken on it was.
                       ref_qwen_short_edge=0)
        p = bench / fname
        written.append((f"{task}-stamped", p, wf))

    def flush():
        """Write every graph. Called only once nothing has objected.

        Writes used to happen inline, as each graph was built, and validation
        ran afterwards over what was already on disk. So a red build still
        SHIPPED its graphs and merely reported a nonzero exit -- which is how
        `MiniMaxH3AppendRefImage` corrupted 40 UI graphs on 2026-08-25 and
        stayed corrupted: the failure was printed every time anyone
        regenerated, and the bad files were already written by then.

        Nothing is written now until the validators and the staleness verdict
        have both passed. A failed build leaves the tree
        exactly as it found it.
        """
        for _t, path, doc in written:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
            print(f"wrote {path.name}")

    if args.no_validate:
        flush()
        return 0
    oi = load_object_info(args.object_info)
    errs = []
    for _t, p, wf in written:
        errs += validate_api(wf, oi, p.name)
    # The core-node probe carries core's defaults as a copy; read them back.
    if any(n.get("class_type") == SOL_CORE_NODE
           for _t, _p, wf in written for n in wf.values()):
        errs += core_sol_defaults_drift(oi)
    if errs:
        print("\nvalidation FAILED -- NOTHING WRITTEN, the tree is unchanged:")
        for x in errs:
            print("  " + x)
        return 1
    verdict, details = object_info_is_stale(oi, args.object_info)
    if verdict in ("stale", "blind"):
        if verdict == "stale":
            print("\nREFUSING to report a clean validation: the served schema "
                  "disagrees with this pack's code on disk, so the graphs "
                  "above were checked against a schema that is not the one "
                  "they will run under. Restart ComfyUI and run this again.")
        else:
            print("\nREFUSING to report a clean validation: this pack's own "
                  "schema could not be read, so whether the server is stale "
                  "was never established. The graphs above were checked "
                  "against a schema nothing confirmed.")
        for x in details:
            print("  " + x)
        return 1
    flush()
    note = ("served schema matches this pack's code on disk" if verdict == "matches"
            else details[0])
    print(f"\nvalidated {len(written)} graphs against {args.object_info}: ok "
          f"({note})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
