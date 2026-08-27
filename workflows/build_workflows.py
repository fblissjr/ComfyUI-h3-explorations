#!/usr/bin/env python3
"""Generate the MiniMax H3 test workflows, in API format and UI format.

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

    uv run --active --no-sync python build_workflows.py

It writes the JSON next to itself and validates every API graph against a
live ComfyUI's /object_info (or a cached copy passed with --object-info).
Validation is static -- nothing is submitted, nothing touches the GPU.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
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
    "MiniMaxH3SageAttention", "SageChainAssert", "MiniMaxH3KeyframeCanvas",
    "MiniMaxH3ReferenceFit", "MiniMaxH3Resolution", "MiniMaxH3Preflight",
    "MiniMaxH3ProvenanceStamp",
}

# Model names, sampler settings, canvas geometry and the SolAttn knobs all
# used to live here in duplicate with the bench. Single source is
# h3_config.py -- see its docstring for why that matters.
from h3_config import (  # noqa: E402
    ENCODER_V2, ENCODER_INT8, CORE_LOADED_ENCODERS, IMAGE_VAE, IMAGE_EDIT_BUDGET,
    ASPECTS, CANVAS, FPS, LENGTH, LONG_LENGTH, MODELS,
    SAMPLING, SAGE_NODE, SEED, SIGMA_SHIFT, SOL_RECOMMENDED_CUDA,
    CACHE_NODE, CACHE_NODE_CLASS,
    TURBO_LORA, TURBO_LORA_STRENGTH, TURBO_SHIFT, TURBO_STEPS,
    TURBO_768P_LORA, TURBO_768P_SHIFT, TURBO_768P_STEPS,
    TURBO_768P_STRENGTH, TURBO_768P_DISTILLED_STEPS,
    turbo_label,
    TURBO_SLA_LORA, TURBO_SLA_SHIFT, TURBO_SLA_STEPS,
    TURBO_OWNER_STRENGTH, TURBO_OWNER_SCHEDULER,
    TURBO_HOME_CANVAS, TURBO_SAMPLER, SPLIT_AT, REF_VIDEO_BUDGET,
    CAPTURE_REF_IMAGES,
    TURBO_PACK_LORA, TURBO_PACK_STEPS, TURBO_PACK_STRENGTH,
    TURBO_PACK_SCHEDULER, TURBO_PACK_LOW_VRAM,
    DIALOGUE_REF_IMAGES,
    SOL_END_PERCENT_BY_STEPS,
    TURBO_REF2VA_LORA, TURBO_REF2VA_STEPS, TURBO_REF2VA_SHIFT,
    PDD_FL2VA_LORA, PDD_REF2VA_LORA, PDD_STEPS, PDD_STEPS_FAST,
    PDD_STRENGTH,
)




# The Sol-Attn node every graph wires. Switched from kijai's Triton pack
# (`SolAttnPatch`) to the CUDA one on 2026-08-14; see SOL_RECOMMENDED_CUDA in
# h3_config.py for what that does and does not carry over.
#
# It is a node id in saved graphs, so it obeys the one rule in CLAUDE.md: the
# UI form matches `widgets_values` POSITIONALLY against the schema, so the
# widget order below must stay in the node's declared input order, widgets
# only (`model` is a socket, not a widget). Verified against a live
# /object_info, which is the only thing that can confirm it.
SOL_NODE = "SolAttnMiniMax"

# `selection` is a DynamicCombo (v3 node, 2026-08-22): choosing an option adds
# that option's own inputs to the node, and the two graph forms encode them
# DIFFERENTLY, which is the whole reason this lives in one place.
#
#   UI form   the option's widgets are spliced in immediately after the
#             selector, not appended at the end. Source read, not a build:
#             ComfyUI_frontend v1.49.6 (the version installed here),
#             `src/core/graph/widgets/dynamicWidgets.ts` --
#             `insertionPoint = node.widgets.findIndex(w => w === widget) + 1`
#             followed by `node.widgets.splice(insertionPoint, 0, ...)`.
#   API form  the option's inputs are keyed under the combo's id with a dot,
#             `selection.tau`, and ComfyUI regroups them into the dict the
#             node receives (`comfy_api/latest/_io.py::build_nested_inputs`).
#             Confirmed against ComfyUI's own validator, which rejects a bare
#             `tau` with "Required input is missing / tau".
#
# **That validator is not a gate.** A graph carrying NO `selection` at all
# validates clean and then dies at execute on `selection["selection"]`, so a
# stale Sol node reaches the queue before anything complains. Regenerate;
# do not hand-edit.
SOL_SELECTION_INPUTS = {
    "adaptive tau": ("tau",),
    "top-k (SLA)": ("keep_percent",),
}
# Widgets after the selection group, in the node's declared input order
# (`model` is a socket, not a widget).
SOL_TAIL_WIDGETS = ("start_percent", "end_percent", "min_tokens",
                    "sink_conditioning", "morton", "morton_curve",
                    "centroid_tail", "reuse_qkv_memory", "verbose",
                    "dense_blocks")


def sol_widget_order(sol):
    """Widget ids in the order the frontend lays them out, for this selection.

    Not a constant, because the middle of the list depends on `selection`.
    """
    try:
        nested = SOL_SELECTION_INPUTS[sol["selection"]]
    except KeyError:
        raise KeyError(f"Sol config selection {sol.get('selection')!r} is not "
                       f"one of {sorted(SOL_SELECTION_INPUTS)}") from None
    return ("selection",) + nested + SOL_TAIL_WIDGETS


def _sol_widgets(sol):
    """Widget values in schema order. Raises rather than emitting a short list.

    A missing key would silently shift every later widget by one, which is
    exactly the failure that cost a real bug on 2026-08-10 -- a saved graph
    stores widgets_values as a bare list and matches by index.
    """
    order = sol_widget_order(sol)
    missing = [k for k in order if k not in sol]
    if missing:
        raise KeyError(f"Sol config is missing {missing}; widgets are positional "
                       f"and a short list re-points every later one")
    return [sol[k] for k in order]


def _sol_title(sol, sol_enabled):
    """Node title, naming `end_percent` when it was derived rather than default.

    The value is picked by the generator from the step count and cannot be
    recomputed by the node -- it converts percent to sigma at patch time, before
    the scheduler downstream has said how many steps there will be. So the
    widget is static and the only place a reader can learn it was derived is
    here.
    """
    if not sol_enabled:
        return "Patch Sol-Attn (bypassed)"
    end = sol.get("end_percent")
    base = SOL_RECOMMENDED_CUDA.get("end_percent")
    if end is None or end == base:
        return "Patch Sol-Attn"
    return (f"Patch Sol-Attn (end_percent {end:g} - derived from the step "
            f"count so the LAST step stays dense; default is {base:g})")


def sol_for_steps(sol, steps):
    """`sol`, with `end_percent` lowered so the final sampling step runs dense.

    A graph's step count decides which steps fall inside Sol's sigma band, so
    the band has to move with it or the dense tail disappears -- which is what
    happened when the PDD arms halved the steps. See
    `h3_config.SOL_END_PERCENT_BY_STEPS` for the derivation and why this one is
    ours rather than the vendor's.

    Untouched at step counts not in the table: 16 and 20 already put their last
    step below the band, so there is nothing to restore and nothing to pay for.
    """
    if sol is None:
        return None
    end = SOL_END_PERCENT_BY_STEPS.get(steps)
    return sol if end is None else dict(sol, end_percent=end)


def sol_api_inputs(sol):
    """API-form inputs: the selected option's inputs are dotted under it.

    Also refuses a config carrying an input that belongs to the OTHER option.
    Such a key would be emitted as an undotted top-level input, which the node
    does not declare and which ComfyUI would reject only at queue time.
    """
    nested = set(sol_widget_order(sol)[1:len(SOL_SELECTION_INPUTS[sol["selection"]]) + 1])
    foreign = {k for opt, keys in SOL_SELECTION_INPUTS.items()
               for k in keys if k in sol} - nested
    if foreign:
        raise KeyError(f"Sol config selects {sol['selection']!r} but also carries "
                       f"{sorted(foreign)}, which belongs to another selection")
    return {(f"selection.{k}" if k in nested else k): v for k, v in sol.items()}


# Prompts for the long presets (362 frames, 15.083s). A 15s request needs a
# shot timeline, not one continuous beat -- the guide wants numbered shots with
# explicit cut times past a few seconds, and a 15s request against a 6s prompt
# leaves the model twelve seconds it was never told about.
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
LONG_T2V_PROMPT = """integrated_multimodal_description:
[Shot 1] Live-action, cinematic, handheld, shallow depth of field. A medium-wide shot frames a covered market aisle in late morning, crates of citrus stacked along a wooden stall front, dust turning slowly in a shaft of light from the roof vents. A stallholder in her fifties with a warm, gravelly alto (S1) sets a crate on the counter, wipes both palms down her apron, and says: <d>[English] Last of the good ones. After this it is all imports.</d> Her lips close and she pushes the crate forward with the heel of her hand. The camera tracks left at medium amplitude and moderate speed as a young porter (S2) steps into frame behind her shoulder.
[Shot 2] At 00:04.500, the shot cuts to a close shot over the porter's shoulder as he squats, takes the crate at its corners, and lifts it to his chest. The porter, a lean man in his twenties with a quick, bright tenor (S2), answers: <d>[English] Then I will take two.</d> His lips close and he shifts the weight onto his hip, and coins clatter one after another into a metal tin on the counter.
[Shot 3] At 00:09.000, the camera whip pans to a wide shot of the aisle as he carries both crates away between the stalls, shoppers stepping aside around him, while the stallholder turns back and stacks fruit into a pyramid with both hands, her lips closed.

overall_soundscape:
Loose crowd murmur under a high roof, wooden crates knocking hollow as they stack, coins dropping one by one into a metal tin, boot steps on swept concrete, and the dry crackle of paper bags shaken open.

non_diegetic_music:
N/A"""

# Sung lines: the lyrics markers WRAP one or more `<d>` blocks rather than
# replacing them.
T2V_REHEARSAL_PROMPT = """integrated_multimodal_description:
[Shot 1] Live-action, cinematic, static tripod framing with motivated practical light and fine film grain. A medium shot frames a basement rehearsal room, foam panels on the walls, a drum kit lit by one clamp lamp. A drummer in his thirties with a low, unhurried baritone (S1) rests his sticks across the snare, looks off toward the vocal booth, and says: <d>[English] From the second verse. Count me in this time.</d> His lips close and he rolls his shoulders once and resets his grip.
[Shot 2] At 00:05.000, the shot cuts to a close shot of a singer in the booth behind glass, headphones on one ear. The singer, a woman in her twenties with a clear, slightly breathy mezzo (S2), leans to the microphone and sings: <|lyrics_start|><d>[English] I walked the long way home again.</d><d>[English] The lights were out on Seventh Street.</d><|lyrics_end|> Her lips close on the last word and she lifts her chin as the drummer's first hit lands.
[Shot 3] At 00:10.000, the camera pushes in with small amplitude at slow speed on the kit as both hands work, the hi-hat opening and closing, one cymbal ringing longer than the rest while the clamp lamp shakes very slightly on its arm.

overall_soundscape:
A snare rimshot and hi-hat chatter in a small treated room, a cymbal ringing down and cut short by a hand, headphone spill leaking thinly, a chair leg scraping once on concrete, and close breathing between phrases.

non_diegetic_music:
N/A"""

# On-screen text and a line truncated by the end of the clip. `<|cutoff|>` sits
# directly against the closing `</d>` with no full stop between -- a full stop
# before it is dragged into `.<` by BPE on an unpatched tokenizer, so the
# spacing is load-bearing (`docs/comfyui_vendor_gaps.md`).
T2V_CLINIC_PROMPT = """integrated_multimodal_description:
[Shot 1] Live-action, cinematic, handheld with restrained depth of field and cool overhead fluorescent light. A medium-wide shot frames a night pharmacy counter, shelves receding behind it, a queue rope empty. <|caption_start|>24 HOUR DISPENSARY<|caption_end|> A pharmacist in her forties with a calm, level contralto (S1) turns a paper bag on the counter so the label faces the customer and says: <d>[English] Twice a day. Not on an empty stomach.</d> Her lips close and she folds the top of the bag over twice, pressing the crease flat with two fingers.
[Shot 2] At 00:04.000, the shot cuts to a close shot of an older customer with a thin, tired tenor (S2) leaning on the counter with both forearms. He takes the bag, turns it once to read it, and asks: <d>[English] And if I forget one?</d> His lips close and he tucks the bag into his coat pocket.
[Shot 3] At 00:08.500, the camera trucks right at small amplitude and slow speed as he walks toward the sliding door, which parts for him and lets in a wash of colder blue light. Behind him the pharmacist slides a drawer shut with her hip and calls after him: <d>[English] And if the swelling does not go down by Thursday you should</d><|cutoff|>

overall_soundscape:
A paper bag folded and pressed flat, a drawer rolling shut and latching, keyboard keys in short bursts, a sliding door motor starting and stopping, and unhurried footsteps on hard flooring.

non_diegetic_music:
N/A"""


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
T2V_SUBWAY_PROMPT = """integrated_multimodal_description:
[Shot 1] Live-action, cinematic, handheld with fast reframing and shallow depth of field under cool platform fluorescents. A wide shot frames a crowded underground platform, tiled columns receding, a train already braking into frame from the right with visible speed. Tiled signage runs above the busker's head. <|caption_start|>NORTHBOUND - PLATFORM 2<|caption_end|> A busker in her twenties with a bright, slightly raw mezzo (S1) stands over an open guitar case, strums once, and sings into the arriving noise: <|lyrics_start|><d>[English] Nobody waits on the northbound line.</d><d>[English] Everybody's leaving on time.</d><|lyrics_end|> Her lips close on the last word as the train's headlights wash across her face and commuters surge past her in both directions, coats swinging, one man breaking into a run.
[Shot 2] At 00:03.500, the shot cuts to a tight two-shot of two commuters shouldering through the crowd at speed, the camera tracking with them at large amplitude. A woman in a soaked raincoat with a clipped, urgent contralto (S2) turns her head without slowing and says: <d>[English] Do not stop, it is the last one.</d> Her lips close. Her companion, a man in his thirties with a breathless, higher tenor (S3), answers half a step behind her: <d>[English] I know, I know, go, go.</d> His lips close and he shoves his bag under one arm as they cut left around a column.
[Shot 3] At 00:07.000, the camera whip pans to a low wide shot of the platform edge as the doors open and the crowd compresses inward, bodies turning sideways, a dropped umbrella skidding across the tiles. The busker keeps playing through it, and over the crowd she sings again: <|lyrics_start|><d>[English] Hold the door and hold your line.</d><|lyrics_end|> Her lips close.
[Shot 4] At 00:11.000, the shot changes to a close shot inside the carriage looking out through the closing doors, the woman in the raincoat pressed against the glass, breathing hard, as she calls back to her companion still on the platform: <d>[English] Get the next one and meet me at the</d><|cutoff|>

overall_soundscape:
Brake squeal rising and cutting out as a train settles, a dense crowd shuffling and coats brushing, a single guitar strummed hard over the noise, an umbrella skittering across tile, a two-tone door chime, and pneumatic doors sealing with a hard thump.

non_diegetic_music:
N/A"""

T2V_KITCHEN_PROMPT = """integrated_multimodal_description:
[Shot 1] Live-action, cinematic, handheld, fast reframing, hard practical light off stainless steel with fine film grain. A medium-wide shot frames a restaurant line mid-service, four burners lit, steam crossing the lens, a ticket rail loaded above the pass. A printed ticket hangs closest to camera. <|caption_start|>TABLE 12 - 2 COVERS - FIRE<|caption_end|> An expediter in his fifties with a hard, carrying baritone (S1) slaps the rail and calls down the line: <d>[English] Two on twelve, fire it now.</d> His lips close and he snaps the ticket free with two fingers. The camera tracks right at large amplitude and fast speed past three cooks working, one tossing a pan so the flame climbs above the rim.
[Shot 2] At 00:03.500, the shot cuts to a close shot of a young line cook with a light, quick soprano (S2) at the flat top, moving fast, who answers without looking up: <d>[English] Two on twelve, heard.</d> Her lips close, and then she sings along under her breath with a radio on the shelf behind her: <|lyrics_start|><d>[English] Keep it moving, keep it hot.</d><|lyrics_end|> Her lips close as she flips two portions in a single motion and the flame flares up behind her shoulder.
[Shot 3] At 00:07.500, the camera pushes in fast with large amplitude on the pass as plates land in a row, hands entering frame from three directions, tongs setting garnish, a thumb wiping a rim clean. The expediter and the cook overlap at speed: <d>[English] Where is my second plate.</d> <d>[English] Behind you, behind you.</d> Both sets of lips close as a plate is spun into position.
[Shot 4] At 00:11.500, the shot changes to a low shot as a runner lifts both plates and turns for the door, the kitchen receding behind him in a blur of steam, while the expediter calls after him already reading the next ticket: <d>[English] And tell them the special is</d><|cutoff|>

overall_soundscape:
A ticket printer chattering in bursts, a metal rail slapped flat, pans ringing on a flat top with sharp oil crackle, a gas burner whumping as it catches, plates set down hard in quick succession, and a thin radio behind everything.

non_diegetic_music:
N/A"""


T2V_SCENES = {
    "market": LONG_T2V_PROMPT,
    "rehearsal": T2V_REHEARSAL_PROMPT,
    "clinic": T2V_CLINIC_PROMPT,
    "subway": T2V_SUBWAY_PROMPT,
    "kitchen": T2V_KITCHEN_PROMPT,
}


def scene_prompt(name: str, *, first_frame: bool = False,
                 last_frame: bool = False) -> str:
    """A baseline scene rendered for the task its sockets describe.

    The t2v and keyframe paths share one node and one three-field layout, so a
    scene is written once and ANCHORED here rather than written twice. What
    changes is only what the description promises about the wired frames:

      first_frame  the guide's preamble line, plus [Shot 1] holding the
                   picture's framing, lighting and composition.
      last_frame   the final shot's composition CONVERGING on the picture at
                   the end, which is the keyframe guide's own wording.

    Label numbering follows what is wired, not what is authored: with both
    frames the last is `<Picture 2>`, with only a last frame it is
    `<Picture 1>`, because the tokenizer numbers the labels the graph emits and
    a prompt naming `<Picture 2>` on a one-picture graph is a dangling label.

    **ref2va is deliberately NOT here.** It is a different six-field layout
    with `<Subject N>` definitions and a retention analysis, built by
    `_ref_prompt()` and checked by `bench/check_ref_prompt_labels.py`. Folding
    it in would mean this function silently emitting the wrong format for a
    node that would still render.
    """
    text = T2V_SCENES[name]
    if not (first_frame or last_frame):
        return text

    last_label = "<Picture 2>" if first_frame else "<Picture 1>"
    if first_frame:
        text = text.replace(
            "[Shot 1] ",
            "[Shot 1] Holding the exact framing, lighting, wardrobe and "
            "composition established in <Picture 1>, ", 1)
    if last_frame:
        # Into the LAST shot, which is the last [Shot N] line before the
        # soundscape field -- appended to that line, not to the field.
        head, sep, tail = text.partition("\n\noverall_soundscape:")
        lines = head.rstrip().split("\n")
        lines[-1] += (f" The camera position, subject placement, wardrobe and "
                      f"exact final composition converge on {last_label} at "
                      f"the end.")
        text = "\n".join(lines) + sep + tail

    # THE ALIGNMENT LINE IS PER-MODE AND THE THREE ARE NOT INTERCHANGEABLE.
    # `base_en.md:14-32` gives one string per task and they differ in more than
    # wording: FL2VA carries NO angle brackets and NO square brackets, where
    # I2VA and L2VA both bracket, and T2VA has no line at all. Until 2026-08-22
    # this function prepended the I2VA sentence whenever `first_frame` was set,
    # so a first+last call emitted the I2VA line for an fl2va task and a
    # last-only call emitted no line at all -- which `preflight_graph.grade`
    # fails outright as a keyframe socket with no preamble.
    #
    # Nothing caught either, and nothing would have: preflight checks that the
    # preamble NAMES a Picture, not that it is the right sentence for the mode.
    # This function is still uncalled (the shipped graphs run on the prompt
    # constants above), so the defect never reached a graph -- but it was
    # staged for exactly the task that would have hit it first.
    if first_frame and last_frame:
        line = ("How the reference pictures align with the target video \u2014 "
                "Picture 1 (from Shot 1) aligns with the 0.00-second mark of "
                f"the target video; Picture 2 (from Shot {FL2V_FINAL_SHOT}) "
                "aligns with the S.SS-second mark of the target video.")
    elif first_frame:
        line = ("For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced.")
    else:
        line = ("How the reference pictures align with the target video \u2014 "
                "<Picture 1> (from [Shot N]) aligns with the S.SS-second mark "
                "of the target video.")
    return line + "\n\n" + text

# h264-mp4 rather than h265 or an nvenc variant: software x264 at crf 19 is
# the most portable mp4 there is, and the nvenc paths trade quality per bit
# for encode speed on a file that takes seconds to write next to a render
# that takes minutes. Switch to video/h265-mp4 if size matters more than
# playing everywhere.
VIDEO_FORMAT = "video/h264-mp4"

# Placeholder input filenames. These are whatever the local install happens
# to have; swap them for your own before running an i2v or r2v graph.
# A reference VIDEO is an IMAGE batch, not a VIDEO: `ref_videos.ref_video_0`
# takes frames. VHS_LoadVideo is the loader because it is the one that exposes
# `force_rate`, and force_rate=24 is not optional here. ComfyUI's node has no
# fps input at all and assumes 24 twice over -- for the DiT's temporal clock
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
# In the input ROOT, not `h3_refs/`: VHS_LoadVideo's `video` widget is a combo
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

# (LoadImage id, MiniMaxH3ReferenceFit id) per reference slot, in socket order.
#
# **Fixed per slot rather than allocated in a loop.** `bench_e2e_h3.py` and
# `bench_image_edit_refs.py` both address the first pair as "15"/"24" by name,
# so a renumbering would silently point a bench at the wrong node. Slot 3 takes
# 34/35 because 26-33 and 40-43 are already spoken for in this graph.
#
# Slots 4-6 were added 2026-08-18 for the workload-grid count ladder. Typed
# append ids are allocated separately below, so these pairs remain only the
# stable loader/fit ids benches address. Slots 4-6 take 36-39 and 45-46: 26-33
# and 40-43 are spoken for (reference loaders, split path, plain chain), and 44
# is the cache node.
_REF_IMAGE_NODES = (("15", "24"), ("16", "25"), ("34", "35"),
                    ("36", "37"), ("38", "39"), ("45", "46"))

# Typed reference append nodes, in presentation order. The six image slots
# above plus one video and one standalone audio reference can consume all
# eight. These ids are outside the long-standing 1-47 API graph allocation so
# the migration does not renumber nodes that benches address directly.
_REF_APPEND_NODES = tuple(str(i) for i in range(50, 58))


def _graph_dir(out, extra: dict):
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
    return out / "image" if extra.get("single_frame") else out


def _ref_image_slots(ref_images_on: bool, ref_image_count: int,
                     ref_images: tuple[str, ...] | None):
    """[(load_id, fit_id, filename)] for the reference images a graph wires.

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
            f"{len(_REF_IMAGE_NODES)} stable loader/fit slots and the same "
            "number of image positions in its typed append-id budget.")
    return [(ld, fit, f) for (ld, fit), f in zip(_REF_IMAGE_NODES, files)]

T2V_PROMPT = """integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a lone lighthouse keeper on a wet stone balcony at dawn, wearing a heavy oilskin coat, the lamp housing glowing behind him. Grey-blue sea fog rolls past below the railing and gulls cross the frame. The camera pushes in with small amplitude at slow speed as he raises a brass telescope, holds it steady against his eye, then lowers it and turns toward the light. [Shot 2] At 00:03.000, the shot cuts to a close-up of the rotating lamp assembly, the beam sweeping past the lens and out into the fog.

overall_soundscape: A low sea swell breaks against stone under a steady wind, with gulls calling overhead. A distant foghorn sounds twice, and the lamp mechanism turns with a slow mechanical grind.

non_diegetic_music: Sustained low strings at a slow tempo with a single sparse piano figure, holding without a swell."""

I2V_PROMPT = """For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the subject shown in <Picture 1> holds its position, framing, lighting, and colors exactly as established in the image. The camera pushes in with small amplitude at slow speed while the subject begins to move, the surrounding scene staying continuous with the reference frame.

overall_soundscape: Quiet room tone with a low ambient hum continues throughout, joined by soft physical sounds from the subject's movement.

non_diegetic_music: N/A"""

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
    two placeholders the other two modes do not.** `base_en.md:24` gives the
    string with `Shot N` and `S.SS` in it; N is the index of the actual final
    shot and S.SS is the effective duration to exactly two decimals. Typing a
    duration here would be a number that silently disagrees with the graph the
    moment `length` changes, so it is derived from the snapped frame count --
    the same grid `MiniMaxH3Conditioning` applies.

    **Note the punctuation.** FL2VA is the one alignment sentence of the three
    that carries no angle brackets and no square brackets: `Picture 1 (from
    Shot 1)`, not `<Picture 1> (from [Shot 1])`. I2VA and L2VA both bracket.
    `base_en.md:14-32` gives all three and this differs from its neighbours by
    exactly that, which is how a writer borrowing the I2VA form gets it wrong
    and nothing goes red -- preflight checks that the preamble names a Picture,
    not that it is the right sentence for the mode.

    One shot, deliberately. `base_en.md:60` says FL2VA "generally favors a
    single shot so the model can interpolate continuously from the first frame
    to the last", and that multiple shots are for when they are explicitly
    specified. So N is 1 here, and stays 1 unless the body grows a cut.
    """
    seconds = duration_of(snap_length(length))
    return (
        "How the reference pictures align with the target video \u2014 Picture 1 "
        "(from Shot 1) aligns with the 0.00-second mark of the target video; "
        f"Picture 2 (from Shot {FL2V_FINAL_SHOT}) aligns with the {seconds:.2f}-second "
        "mark of the target video.\n"
        "\n"
        "integrated_multimodal_description: [Shot 1] Live-action, cinematic, one "
        "continuous shot that begins in the exact state of the opening frame and "
        "ends in the exact state of the closing frame. The subject holds the "
        "opening position, framing, lighting and colors, then moves steadily "
        "through the space while the camera trucks right with small amplitude at "
        "slow speed. Wardrobe, palette and the surrounding scene stay continuous "
        "throughout, and the subject's pose, placement and the camera's position, "
        "angle and framing converge on the closing composition, reaching it only "
        "at the final frame.\n"
        "\n"
        "overall_soundscape: Quiet room tone with a low ambient hum continues "
        "throughout, joined by soft physical sounds from the subject's movement "
        "and a single settling sound as the motion comes to rest.\n"
        "\n"
        "non_diegetic_music: N/A")


R2V_PROMPT = """subject_definitions:
<Subject 1> is the main character in <Picture 1>, whose face, hair, and clothing are carried into the target video.
<Subject 2> is the environment in <Picture 2>, which provides the setting for the target video.

summary:
[reference generation] The target video places <Subject 1> inside <Subject 2> for a single continuous shot.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - face, hair, and clothing are retained.
<Subject 2> (appears in [Shot 1]): fully_preserved - the visual setting is retained.

detailed_description:
The target video is in a cinematic live-action style with realistic natural textures, depth of field, and continuous motion throughout the ten-second sequence.
[Shot 1] The shot opens on a wide-angle establishing perspective of <Subject 2>, framing the scene with organic spatial depth and clean perspective lines under the ambient light of the environment. From 0.0s to 2.5s, <Subject 1> enters steadily into frame from camera-left, stepping forward in full view while preserving the complete identity, form, and wardrobe specified by <Picture 1>. From 2.5s to 6.0s, the camera executes a controlled, slow-speed lateral truck to the right on a smooth dolly axis, tracking parallel to <Subject 1> as they reach the center of the frame and pause in a natural resting posture. During this middle interval, the lens maintains sharp optical focus on <Subject 1> while <Subject 2> recedes with gentle focal roll-off and accurate geometric parallax across the background layer. From 6.0s to 8.5s, <Subject 1> turns subtly toward the primary ambient light source of the environment, glancing slightly off-camera with a measured, contemplative expression, allowing their profile and form to be clearly visible without abrupt shifts in posture. From 8.5s to 10.0s, the camera holds steady on a balanced medium shot framing <Subject 1> against <Subject 2>, preserving subject presence, lighting equilibrium, and environmental continuity until the final frame without any camera cuts or sudden movements.

overall_soundscape:
Natural ambient atmosphere continues throughout the shot.

non_diegetic_music:
N/A"""


# --------------------------------------------------------------------------
# API format
# --------------------------------------------------------------------------

sys.path.insert(0, str(HERE.parent))
from h3_rules import (  # noqa: E402
    aspect_in_range, describe_aspect_range, describe_length,
    duration_in_range, duration_of, is_single_frame, max_legal_length,
    min_legal_length, snap_length,
)


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
    # validates over HTTP -- so put the root on the path just for this.
    root = HERE.parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
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


def _plain_model_chain(g, *, sage, sol, shift, head_chunks):
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
    g["43"] = {"class_type": "SageChainAssert",
               "inputs": {"model": src, "require_override": sage,
                          "require_forward_patch": sage, "exercise": sage,
                          "warn_only": not sage}}
    return ["43", 0]


def build_api(task: str, *, sage: bool = True, prompt: str | None = None,
              length: int = LENGTH, seed: int = SEED,
              sol: dict | None = None, canvas_mode: str = "match_keyframe",
              last_frame: bool = False,
              stamp: bool = False, unet: str | None = None,
              lora: tuple[str, float] | None = None,
              steps: int | None = None, shift: dict | None = None,
              sampler_name: str | None = None, scheduler_name: str | None = None,
              head_chunks: int | None = None, ref_upscale: bool = True,
              ref_video_policy: str = "encoder",
              ref_image_policy: str = "comfy",
              ref_qwen_short_edge: int = 0,
              ref_video: bool = False, ref_video_audio: bool = True,
              ref_images_on: bool = True, ref_image_count: int = 2,
              ref_images: tuple[str, ...] | None = None,
              turbo_pack: bool = False,
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
              ref_audio: bool = False,
              split_at: int | None = None,
              split_base_last: bool = True,
              single_frame: bool = False,
              cache: dict | None = None,
              sla_router: float | None = None,
              vae_encoder: str | None = None,
              clip: str | None = None,
              out_prefix: str | None = None, **canvas) -> dict:
    """API-format graph, submittable as {"prompt": <this>} to POST /prompt.

    Node ids match `bench/bench_e2e_h3.py` so a timing run and a hand-edited
    graph can be compared node-for-node; "10" is the sampler in every graph.

    `unet` overrides the checkpoint the task would otherwise pick, for the
    probes that need a model source no task name describes. `lora` is
    (name, strength) and inserts a LoraLoaderModelOnly.
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
    # THE DEFAULT PROMPT FOLLOWS THE SOCKETS, NOT THE TASK STRING. `i2v`
    # covers both keyframe modes -- one wired frame or two -- and they take
    # DIFFERENT alignment sentences (`base_en.md:14-32`), so keying this on
    # `task` alone hands an fl2va graph the I2VA line. Nothing downstream
    # would catch it: preflight checks that the preamble names a Picture,
    # not that it is the right sentence for the mode.
    if prompt is None:
        prompt = (fl2v_prompt(length) if (task == "i2v" and last_frame)
                  else {"t2v": T2V_PROMPT, "i2v": I2V_PROMPT,
                        "r2v": R2V_PROMPT}[task])

    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet or MODELS["unet_ref2va" if ref else "unet_fl2va"],
                         "weight_dtype": "default"}},
        # Core CLIPLoader lists every file in the directory but cannot open a
        # compressed-tensors W4A16 artifact; the repo adapter opens only those.
        # The file decides the loader (h3_config.CORE_LOADED_ENCODERS).
        "2": ({"class_type": "CLIPLoader",
               "inputs": {"clip_name": clip, "type": "minimax", "device": "default"}}
              if clip in CORE_LOADED_ENCODERS else
              {"class_type": "MiniMaxH3AWQEncoderLoader",
               "inputs": {"encoder_name": clip or MODELS["clip"],
                          "device": "default"}}),
        # The image VAE ONLY on the single-frame path. See h3_config: same
        # frozen encoder, decoder retrained for one temporal latent, and its
        # own README says it regresses multi-frame reconstruction -- so this
        # swap must never be reachable from a graph that renders a clip.
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": IMAGE_VAE if single_frame
                         else MODELS["video_vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["audio_vae"]}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        # The turbo pack ships its own SAMPLER source rather than a name for
        # KSamplerSelect. On a recent ComfyUI it self-reports as bit-for-bit
        # the stock result -- it exists to keep older builds stepping the
        # audio stream on its own clock, which is precisely the thing that
        # breaks first at low step counts, and every reference arm here
        # carries audio.
        "7": ({"class_type": "MiniMaxH3TurboSampler", "inputs": {}}
              if turbo_pack else
              {"class_type": "KSamplerSelect",
               "inputs": {"sampler_name": sampler_name or SAMPLING["sampler"]}}),
        "8": {"class_type": "BasicScheduler",
              "inputs": {"model": None,
                         "scheduler": scheduler_name or SAMPLING["scheduler"],
                         "steps": steps if steps is not None else SAMPLING["steps"],
                         "denoise": SAMPLING["denoise"]}},
        "9": {"class_type": "BasicGuider",
              "inputs": {"model": None, "conditioning": ["5", 0]}},
        "10": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0],
                          "sigmas": ["8", 0], "latent_image": ["5", 1]}},
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
        g["5"] = {"class_type": "MiniMaxH3ReferenceConditioning",
                  "inputs": {"clip": ["2", 0], "vae": vae_enc,
                             "audio_vae": ["4", 0],
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
        for load_id, _fit_id, fname in slots:
            g[load_id] = {"class_type": "LoadImage", "inputs": {"image": fname}}
        chain = None
        append_ids = iter(_REF_APPEND_NODES)
        for load_id, _fit_id, _fname in slots:
            append_id = next(append_ids)
            append_inputs = {"image": [load_id, 0], "size_policy": "max"}
            if chain is not None:
                append_inputs["references"] = chain
            # Appended inputs go after `references`, matching the schema order
            # a saved graph maps widget values by.
            append_inputs["allow_upscale"] = ref_upscale
            append_inputs["short_edge"] = _ref_short_edge()
            # Only when set: 0 is the node's default and the byte-for-byte
            # one-view path, and writing it into every graph would touch every
            # shipped reference graph for nothing.
            if ref_qwen_short_edge:
                append_inputs["qwen_short_edge"] = ref_qwen_short_edge
            g[append_id] = {"class_type": "MiniMaxH3AppendRefImage",
                            "inputs": append_inputs}
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
            g["28"] = {"class_type": "VHS_LoadVideo",
                       "inputs": {"video": PLACEHOLDER_VIDEO,
                                  "force_rate": REF_VIDEO_FORCE_RATE,
                                  "custom_width": 0, "custom_height": 0,
                                  "frame_load_cap": length,
                                  "skip_first_frames": 0,
                                  "select_every_nth": 1, "format": "AnimateDiff"}}
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
        if chain != [terminal_ref, 0]:
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
            g["15"] = {"class_type": "LoadImage", "inputs": {"image": PLACEHOLDER_IMAGE_A}}
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
        # docs/SOLATTN.md's Ordering section, and SageChainAssert, which fails
        # the render when it is violated). A LoRA in front of both is
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
            g["18"] = {"class_type": "MiniMaxH3PDDLoRA",
                       "inputs": {"model": model_src, "lora_name": lora[0],
                                  "strength": lora[1],
                                  "patch_heads": pdd_heads,
                                  "nfe": pdd_nfe}}
        else:
            g["18"] = ({"class_type": "MiniMaxH3TurboLoRA",
                        "inputs": {"model": model_src, "lora_name": lora[0],
                                   "strength": lora[1],
                                   "low_vram": TURBO_PACK_LOW_VRAM}}
                       if turbo_pack else
                       {"class_type": "LoraLoaderModelOnly",
                        "inputs": {"model": model_src, "lora_name": lora[0],
                                   "strength_model": lora[1]}})
        model_src = ["18", 0]
    # Always present, at the base checkpoint's own 12/3, so it changes nothing
    # by default. It is here to be edited: the turbo LoRAs carry their own
    # training shifts (the 768p 4-step wants 6/3), and a graph without this
    # node gives you nowhere to set that and no hint you needed to. Upstream of
    # sage so the sage-then-Sol adjacency below stays intact -- this patches
    # model sampling, which is a different surface from either of them.
    # Node id 19; 18 is the LoRA and 20/21/22 are already spoken for.
    g["19"] = {"class_type": "MiniMaxH3SigmaShift",
               "inputs": {"model": model_src,
                          **(shift if shift is not None else SIGMA_SHIFT)}}
    model_src = ["19", 0]
    if sla_router is not None:
        # The SLA top-k router in place of the sage+Sol chain: a comparative
        # arm (the attention the Turbo-SLA LoRA was distilled under), never a
        # default. Callers pass sage=False and sol=None with it; the assert
        # below then runs warn-only, which is right -- the router takes the
        # call and there is no sage for the chain assert to find. Node id 45:
        # 40-44 are the split chain and the cache.
        if sage or sol is not None:
            raise SystemExit("sla_router is an alternative to the sage+Sol chain; "
                             "pass sage=False and sol=None with it")
        g["45"] = {"class_type": "MiniMaxH3SLARouter",
                   "inputs": {"model": model_src, "sparsity_ratio": sla_router}}
        model_src = ["45", 0]
    if sage:
        g["20"] = {"class_type": "MiniMaxH3SageAttention",
                   "inputs": {"model": model_src, **dict(
                       SAGE_NODE,
                       **({} if head_chunks is None
                          else {"head_chunks": head_chunks}))}}
        model_src = ["20", 0]
    if sol is not None:
        # After sage, never before -- SolAttn composes with the attention
        # patches it finds, and reversed it overwrites ours and you silently
        # get sage only. Node id 21 matches `bench/bench_e2e_h3.py`.
        g["21"] = {"class_type": SOL_NODE,
                   "inputs": {"model": model_src, **sol_api_inputs(sol)}}
        model_src = ["21", 0]
    # Last in the chain, because it asserts what the composition ended up
    # as, not what any one node intended. Sol-Attn negotiates with our
    # override through a duck-typed attribute that both sides rewrote within
    # a minute of each other once already; when that seam breaks the render
    # still succeeds and is quietly slower or numerically different. This
    # turns that into a refused render. `exercise` stays on: install-time
    # evidence is exactly what today has taught us not to trust.
    # `warn_only` follows `sage`: with the node absent this graph is a
    # control arm, and a gate that always raises on the control makes the
    # comparison impossible to run rather than making it safe.
    g["23"] = {"class_type": "SageChainAssert",
               "inputs": {"model": model_src, "require_override": sage,
                          "require_forward_patch": sage, "exercise": sage,
                          "warn_only": not sage}}
    model_src = ["23", 0]

    if cache is not None:
        # Step caching, AFTER the assert: the assert grades the attention
        # composition, and the cache is a forward-skipping wrapper on top of
        # it, not part of it. On a reused step nothing downstream of the
        # wrapper runs -- sage and Sol included -- which is the mechanism, not
        # a conflict. See CACHE_NODE in h3_config.py for why this arm exists
        # and its er_sde caveat. Node id 44: 28-33 are the reference loaders
        # and split path, 34-39 the reference image slots, 40-43 the plain
        # chain `split_at` builds.
        g["44"] = {"class_type": CACHE_NODE_CLASS,
                   "inputs": {"model": model_src, **cache}}
        model_src = ["44", 0]

    # Reports what the assembled conditioning actually costs, before the
    # sampler runs. Pass-through, so it cannot change the render.
    g["26"] = {"class_type": "MiniMaxH3Preflight",
               "inputs": {"conditioning": ["5", 0], "samples": ["5", 1]}}

    # The fork. Both consumers, always, from the same variable.
    g["8"]["inputs"]["model"] = model_src
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
                                       head_chunks=head_chunks)
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
                              "sigmas": ["8", 0], "note": f"bench {task}"}}
        g["11"]["inputs"]["samples"] = ["22", 0]
        g["12"]["inputs"]["samples"] = ["22", 0]
    return g


# --------------------------------------------------------------------------
# UI format
# --------------------------------------------------------------------------

class UIGraph:
    """Minimal litegraph workflow writer.

    Field shapes are copied from the bundled `video_minimax_h3_r2v` template,
    which is the one H3 template that is already flat, so this emits the same
    dialect the frontend just loaded from disk.

    Deliberately no widget-to-input conversions and no helper nodes
    (ResolutionSelector, ComfyMathExpression, PrimitiveStringMultiline). The
    templates use those for convenience, but every one of them is another
    place a hand-edit can go wrong, and the point of these copies is to be
    easy to edit. Resolution, length and prompt are plain widget values on
    the conditioning node.
    """

    def __init__(self):
        self.nodes: list[dict] = []
        self.links: list[list] = []
        self._next_node = 1
        self._next_link = 1

    def add(self, type_: str, pos, *, widgets=None, inputs=None, outputs=None,
            size=(320, 100), title=None):
        nid = self._next_node
        self._next_node += 1
        n = {
            "id": nid, "type": type_, "pos": list(pos), "size": list(size),
            "flags": {}, "order": 0, "mode": 0,
            "inputs": [dict(i) for i in (inputs or [])],
            "outputs": [dict(o) for o in (outputs or [])],
            # Deliberately NOT emitting `cnr_id` or `aux_id`, reversing a
            # change made earlier on 2026-08-11.
            #
            # `cnr_id` lets ComfyUI-Manager offer "install missing custom
            # nodes" to someone who opens this graph without the pack. That
            # audience is strangers pulling from a public registry, which is
            # not how this repo is used: local only, private, LAN remote. So
            # the benefit is near zero here, while `useConflictDetection`
            # ships in the same lazily-loaded chunk as
            # `useComfyRegistryService` (baseURL https://api.comfy.org) and
            # the consuming path was not proven to stay local. Under a
            # local-only constraint, unproven beats unlikely.
            #
            # There is also a squatting edge: we would be claiming
            # "comfyui-h3-explorations", and if a stranger registers that
            # name later, a user's "install missing" click resolves to their
            # package rather than nothing.
            #
            # `aux_id` is worse and must never be added automatically. Its
            # conventional value is the git remote's owner/repo, and this
            # repo's only remote is a LAN address -- deriving it would write
            # a private IP into every shared workflow.
            "properties": {"Node name for S&R": type_},
        }
        if widgets is not None:
            # A dict stays a dict. Most nodes serialize widgets_values as a
            # positional list, but a node whose widget set depends on another
            # widget cannot -- VHS_VideoCombine adds pix_fmt/crf/... after
            # `format`, so position cannot address them and the frontend
            # writes a keyed object instead. `list(a_dict)` silently yields
            # the keys, which is a graph that loads and renders with every
            # setting wrong.
            n["widgets_values"] = (dict(widgets) if isinstance(widgets, dict)
                                   else list(widgets))
        if title:
            n["title"] = title
        self.nodes.append(n)
        return nid

    def _node(self, nid):
        for n in self.nodes:
            if n["id"] == nid:
                return n
        raise KeyError(nid)

    def link(self, src, src_slot, dst, dst_input_name, type_):
        lid = self._next_link
        self._next_link += 1
        s, d = self._node(src), self._node(dst)
        s["outputs"][src_slot].setdefault("links", [])
        if s["outputs"][src_slot]["links"] is None:
            s["outputs"][src_slot]["links"] = []
        s["outputs"][src_slot]["links"].append(lid)
        for inp in d["inputs"]:
            if inp["name"] == dst_input_name:
                inp["link"] = lid
                break
        else:
            raise KeyError(f"{d['type']} has no input {dst_input_name!r}")
        self.links.append([lid, src, src_slot, dst, self._input_index(d, dst_input_name), type_])
        return lid

    @staticmethod
    def _input_index(node, name):
        return [i["name"] for i in node["inputs"]].index(name)

    def _topo_order(self):
        # `order` is advisory -- the frontend recomputes it -- but an
        # inconsistent value shows up as nodes drawn in a nonsense sequence,
        # so emit a real topological order.
        incoming = {n["id"]: set() for n in self.nodes}
        for lid, src, _ss, dst, _ds, _t in self.links:
            incoming[dst].add(src)
        order, placed = {}, set()
        i = 0
        while len(placed) < len(self.nodes):
            progressed = False
            for n in self.nodes:
                nid = n["id"]
                if nid in placed or not incoming[nid] <= placed:
                    continue
                order[nid], i = i, i + 1
                placed.add(nid)
                progressed = True
            if not progressed:
                raise RuntimeError("cycle in graph")
        for n in self.nodes:
            n["order"] = order[n["id"]]

    @staticmethod
    def _uuid_for(name: str) -> str:
        """A stable UUID for a graph, derived from its name.

        The frontend writes `id` as a UUID and we were writing a readable
        slug. Deterministic rather than random so regenerating a graph does
        not churn its identity in git, and so the same graph keeps the same
        id across machines.

        The namespace seed is a bare string rather than a URL. Determinism is
        the only property needed, and the first version seeded from a
        github.com URL that named a handle and a repository -- both wrong,
        and neither anyone's business in a published repo.
        """
        import uuid

        return str(uuid.uuid5(uuid.NAMESPACE_URL,
                              f"comfyui-h3-explorations/{name}"))

    def dump(self, workflow_id: str) -> dict:
        self._topo_order()
        return {
            # Frontend saves carry extra.ds; without it litegraph opens at
            # its default viewport and these graphs start at x = -2860, so
            # the first thing you see is empty canvas.
            "extra": {"ds": {"scale": 0.5, "offset": [3000.0, 400.0]}},
            "id": self._uuid_for(workflow_id), "revision": 0,
            "last_node_id": self._next_node - 1,
            "last_link_id": self._next_link - 1,
            "nodes": self.nodes, "links": self.links, "groups": [],
            "config": {}, "version": 0.4,
        }


def _in(name, type_, *, optional=False, widget=False, label=None):
    d = {"name": name, "type": type_, "link": None}
    if label:
        d["label"] = label
    if optional:
        d["shape"] = 7
    if widget:
        d["widget"] = {"name": name}
    return d


def _out(name, type_):
    return {"name": name, "type": type_, "links": None}


# Text for the in-graph notes. Kept next to the builder rather than in
# docs/h3_geometry_and_nodes.md on purpose: that doc is the long form, this
# is what you need with the graph open. Numbers here come from
# comfy_extras/nodes_minimax_h3.py, not from lore.
_NOTE_GEOMETRY = """\
## You pick an aspect ratio. The resolution follows from it.

`adapt_canvas()` reads your two numbers as a ratio and derives the pixels:
short edge starts at 768, the area caps at 1,032,192 (768x1344), each axis
rounds to 32. Asking for 4K gives the same resolution as 720p at the same
ratio. Exactly 95 resolutions exist across the legal 1/4 to 4 aspect range.

Full table, the derivation, and the length and int32 axes:
`docs/h3_resolutions.md`.

## The fourteen worth knowing

| Ask for | Resolution | Video tokens/frame | Attention |
|---|---|---|---|
| 21:9 | 1536x672 | 1008 | 1.00x |
| 2:1 | 1440x704 | 990 | 0.96x |
| 16:9 | 1344x768 | 1008 | 1.00x |
| 5:3 | 1280x768 | 960 | 0.91x |
| 3:2 | 1152x768 | 864 | 0.73x |
| 4:3 | 1024x768 | 768 | 0.58x |
| 5:4 | 960x768 | 720 | 0.51x |
| 1:1 | 768x768 | 576 | 0.33x |
| 4:5 | 768x960 | 720 | 0.51x |
| 3:4 | 768x1024 | 768 | 0.58x |
| 2:3 | 768x1152 | 864 | 0.73x |
| 9:16 | 768x1344 | 1008 | 1.00x |
| 1:2 | 704x1440 | 990 | 0.96x |
| 9:21 | 672x1536 | 1008 | 1.00x |

All fourteen reproduce themselves, so typing one into width/height gives it
back. 1:1 costs a third of 16:9 at the same frame count, and attention
dominates the step, so this is the largest speed lever anywhere, larger than
any kernel or sparsity setting.

Attention goes as the square of the token count. Video tokens per frame are
`(w//32) * (h//32)`, which is symmetric, so portrait and landscape of a ratio
cost the same. 16:9 against 9:16 is a quality question, never a speed one.

## Where the 32 comes from

The VAE compresses space by 16, then the DiT patchifies that latent 2x2
before attending it. 16 x 2 = 32. Divisible by 16 alone leaves an odd latent
axis the patchify cannot tile.

Core's conditioning nodes do not apply `adapt_canvas()` to the video
resolution at all: width and height are plain ints at step 32, so what you
type is what you get. The 768 and the area cap describe the trained family,
not a limit the node enforces.

## Two things that surprise people

The short edge is not always 768. It is 768 only while the area cap does not
bind, roughly 3:4 through 7:4. Outside that the cap takes over: 21:9 is
1536x672, 9:21 is 672x1536.

1.00x is not the ceiling. Rounding to 32 can land above the 16:9 token count.
Ask for 23:7 and you get 1856x576, which is 1044 video tokens against 1008,
so 1.073x the attention for no extra pixels. That is the worst case in the
set. Nearby ratios behave: 29:9 gives 1824x576 at 1.036x. Stay on the
fourteen unless you have a reason.

## If you want this decided for you

`MiniMax H3 Keyframe Resolution` (this repo) derives the resolution from your
first keyframe the way the reference pipeline does, fits the keyframes onto
it, and reports the cost before you render. The first-frame graph is wired
that way. Text-to-video has no keyframe to derive from, so type a row above.

## Length rounds up to n % 17 == 5

Ask 200, get 209. Ask 300, get 311. Near the top: 311, 328, 345, 362.

362 is the ceiling -- 15.083s, and the longest length H3 was trained on.
ComfyUI's own node accepts up to 3600 with no ceiling at all. Ask for 363 and
you get 379, which is over, and that is why the check runs on the rounded
number rather than the request.

The reference pipeline stops one grid step earlier, at 345, because its
`max_duration` is a hard-coded 15.0s. That is a fact about diffusers, not a
limit on the model: a graph at 362 renders here and will not run unmodified
there. There is no on-grid count at exactly 15.0s, which is how the gap
appears.

At 345 frames attention is ~76% of the step, against ~50% at 124, so long
clips are where sparsity and kernel work pay off most. 362 is 5% longer
again.

The frame count is not the sequence-length ceiling. At 1344x768, 345 frames
is S=108,078 -- already past the fused-layout int32 crossing at 99,864
tokens -- and 362 is longer still. That is safe here only because this repo's
node refuses any sageattention without `sageattn_consume`. See the doc.
"""


_NOTE_NODES = f"""\
## Node order is load-bearing

```
Load Diffusion Model
  -> ModelSamplingMiniMaxH3       (sigma shift; anywhere before the fork)
  -> MiniMax H3 SageAttention     (this repo)
  -> SolAttnMiniMax               (must be AFTER)
  -> BasicScheduler + BasicGuider (MODEL forks to BOTH)
```

**Sol-Attn must come second.** It composes with the attention patch it
finds; reversed, it overwrites ours and you silently get sage only, with no
error and no log line saying so.

**The sigma shift is here to be changed, not because it does anything at
12/3.** Those are the base checkpoint's training shifts, so the node is a
no-op as shipped. The turbo LoRAs inherit the sampler's shift instead of
carrying their own, and the {turbo_label(TURBO_768P_LORA)} one was
distilled at video shift **6** -- and that is the variant trained at 1344x768,
this canvas. Load it without changing this and you sample it off a schedule it
never saw.
Steps move with it too: 16 is a base-model number, these want 4 or 8. The
4-step v0.1 and 8-step v1.0 were both distilled at 12/3 and need no change
here.

**MODEL forks to two consumers.** Rewiring only the guider leaves the
scheduler reading sigmas off the unpatched model, and the render still
succeeds -- which is why that mistake survives.

## Check it is actually running, once per graph change

Turn `verbose` on in Patch Sol-Attn (MiniMax) for one render, then off. You want three
lines. **Read them in the terminal** -- piping or redirecting block-buffers
the output and they may not appear even when everything is fine.

```
sage routing: arch=sm89 ... pv_accum=fp32+fp16 -> fp8_cuda++
[sol_attn] chaining onto an existing attention override
[sol_attn] sparse (1, ..., 56, 128) tau=1.3 int8 pointer
```

Line 1: sage engaged on the fast kernel. Line 3: sparse engaged at your tau.
**Line 2 is the order check** -- it only prints when Sol-Attn finds sage's
override already installed. Missing means the nodes are backwards and you
are paying full price for a render that otherwise looks fine.

## What each node is here for

- **ModelPreviewOverrideKJ** -- taeh3 preview, and it is arguably the
  largest optimization here rather than a convenience. Killing a bad seed at
  90s instead of 11 minutes saves ~9.5 min; the entire kernel and sparsity
  stack saves ~7 min per render. If one render in three is a bad seed the
  preview beats everything else combined -- and they compound rather than
  compete.
- **MiniMax H3 SageAttention** -- INT8-QK / FP8-PV kernel on all 50 DiT
  attention forwards, plus an `optimized_attention_override` registration.
  That second part is what lets Sol-Attn compose instead of bypassing sage.
- **Patch Sol-Attn (MiniMax)** -- block-sparse attention, on the CUDA
  kernel (`comfy_kitchen.sol_attn`). Settings are pinned from
  `workflows/h3_config.py`; edit there and regenerate, not here.

## Deliberately absent

- **MiniMaxH3MemoryEfficientSageAttentionPatch** (KJNodes) -- same job as
  our node, patches the same key, so they conflict. Ours also registers the
  override.
- **MiniMaxLowVRAMAttention** -- head chunking. ~3227 MiB saved at 4 groups
  (measured; three times the ~1070 this note carried before 2026-08-13), but
  1000 attention calls become 4000. On 24GB freed VRAM converts to wall-clock
  at a ~2.6% ceiling. Take it only if you are actually hitting OOM.
- **MiniMaxChunkFeedForward** -- at 362 frames attention peaks ~17.8 GiB
  against FFN's 9-12, so it chunks a peak that is not binding. Short-clip
  feature.
- **PathchSageAttentionKJ** -- global no-guard sage switch. Prefer the
  per-workflow node.
"""

def _probe_note(subject, companion, changed, compare, expect,
                held="same prompt, same canvas"):
    """Note for a probe graph: one variable, its twin, and what to look at.

    A probe that does not name its companion and its seed is a graph with an
    unusual setting, not an experiment. Every one of these is identical to its
    twin except the line under "what differs", and they share
    `h3_config.SEED`, so anything you see between them is that line.

    `held` is what stays fixed, and it is a parameter because the default
    sentence claimed "same prompt" -- which is a contradiction on the two image
    probes whose prompt IS the variable. A boilerplate line that contradicts
    the paragraph under it teaches the reader to skim the boilerplate.
    """
    return f"""\
## Probe: {subject}

**Run this against `{companion}`.** Same seed ({SEED}), {held}, same
everything except one setting. That is the whole design: if the seed moved
between the two, the difference you are looking for would be underneath the
difference you are not.

**What differs:** {changed}

**What to compare:** {compare}

**What to expect:** {expect}

This is a probe, not a render config. If you like what one side does, change
the setting in the shipped graph rather than rendering from this file.
"""


_NOTE_SIZING = """\
## What the sizing nodes decide, and what Preflight tells you

**Preflight is pass-through.** It changes nothing. It reads the assembled
conditioning through the model's own `PackedLayout`, so the sequence length
it draws is the one attention will actually run at.

Read it top to bottom:

```
1152x768  trained family  864 video tokens/frame
124 frames (5.167s)  37 latent frames
sequence length 52,702
  video         31,968  ############........   60.7%
  references    17,216  #######.............   32.7%
  text           3,104  #...................    5.9%
  audio            414  ....................    0.8%
if the aspect ratio changed, same length:
  1:1   768x768      42,046    -20%
  16:9  1344x768     58,030    +10%
```

The percentages are the decision. Reference tokens are attended at every
sampling step exactly as video tokens are, so references at a third of the
sequence means a third of your attention cost is spent describing them.

**"trained family" vs "OUTSIDE trained family".** Core's conditioning nodes
take width and height as plain ints and never call `adapt_canvas`, so the 768
short edge and the 768x1344 area cap constrain nothing you type. 1024x1024 is
legal, renders, costs more per frame than 16:9, and is outside the family the
checkpoint was trained on. Outside is a choice, not an error -- but it should
be one you made on purpose.

## Keyframes and references are not sized the same way

- A **keyframe** is patchified on the video's own latent grid, so its
  resolution must equal the video's. That is why *MiniMax H3 Keyframe
  Resolution* outputs width and height: the keyframe decides them.
- A **reference** is patchified on its own grid, so its resolution only sets
  how many vision tokens it contributes, and *Append Picture* carries that
  decision itself rather than outputting a size.

## Reference sizing lives on the Append Picture node

These graphs use this repo's `MiniMaxH3AppendRefImage`, not native ComfyUI's
autogrow image sockets. Each append carries its own `size_policy`,
`allow_upscale` and `short_edge`, and *MiniMax H3 Reference Conditioning*
performs ONE resize with the target canvas in scope. There is no separate
Reference Resolution node in the chain any more -- it stays registered for
saved graphs that wire it, and its fit is idempotent with this one.

`max` sizes from `short_edge` and keeps the reference on its own grid; `match`
sizes it from the target canvas area instead.

`allow_upscale` on is the released pipeline's behaviour: it scales until the
short side reaches 2048, enlarging small references. ComfyUI's own rule only
ever shrinks. Upscaling adds tokens, not detail, so whether it helps an
already-small source is unmeasured -- turn it off and watch the Preflight
percentages if you want the cheap version.

`image_policy` on the conditioner is a separate decision: it selects WHOSE
still-image ceiling applies once the reference has been prepared. `comfy`
leaves it to whatever processor the loaded CLIP carries, which is the default
and what these graphs have always done. `encoder` and `release` pre-apply the
selected policy's own bounds so the VAE and Qwen stay on one size.
"""


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
#: ref-en.txt:231, and `bench/preflight_graph.py` warns when one is defined and
#: never cited -- so a scene that cannot cite a role is refused rather than
#: emitted with the label missing.
#:
#: Speaker ids stay OUT of retention_analysis; a `(Sx)` there is a hard fail
#: (ref-en.txt:278). They belong here, in the description.
REF_SCENE_SHOTS = {
    "subway": [
        "[Shot 1] Handheld with fast reframing under cool platform fluorescents. "
        "A wide shot establishes {environment_beat}a crowded underground platform, "
        "tiled columns receding, a train braking into frame from the right with "
        "visible speed. <|caption_start|>NORTHBOUND - PLATFORM 2<|caption_end|> "
        "{character} stands over an open guitar case, preserving the face, hair, "
        "wardrobe and build established in the reference, strums once, and sings "
        "into the arriving noise: <|lyrics_start|><d>[English] Nobody waits on the "
        "northbound line.</d><d>[English] Everybody's leaving on time.</d>"
        "<|lyrics_end|> Her lips close on the last word as the headlights wash "
        "across her face and commuters surge past in both directions, one man "
        "breaking into a run.",
        "[Shot 2] At 00:03.500, the shot cuts to a tight two-shot of two commuters "
        "shouldering through the crowd at speed, the camera tracking with them at "
        "large amplitude. A woman in a soaked raincoat with a clipped, urgent "
        "contralto (S2) turns her head without slowing and says: <d>[English] Do "
        "not stop, it is the last one.</d> Her lips close. Her companion, a man in "
        "his thirties with a breathless, higher tenor (S3), answers half a step "
        "behind her: <d>[English] I know, I know, go, go.</d> His lips close and "
        "he shoves his bag under one arm as they cut left around a column.",
        "[Shot 3] At 00:07.000, the camera whip pans to a low wide shot of the "
        "platform edge as the doors open and the crowd compresses inward, a "
        "dropped umbrella skidding across the tiles. {character} keeps playing "
        "through it, her identity unchanged from the reference, and sings over "
        "the crowd: <|lyrics_start|><d>[English] Hold the door and hold your line."
        "</d><|lyrics_end|> Her lips close. The camera holds wide long enough "
        "to keep the tiled columns, the platform edge markings and the "
        "overhead signage of the reference setting continuously visible behind "
        "her while the crowd moves across the frame in both directions, coats "
        "and bags passing close to the lens without occluding her face.",
        "[Shot 4] At 00:11.000, the shot changes to a close shot inside the "
        "carriage looking out through the closing doors, the woman in the raincoat "
        "pressed against the glass, breathing hard, calling back to her companion "
        "still on the platform: <d>[English] Get the next one and meet me at the"
        "</d><|cutoff|>",
    ],
    "kitchen": [
        "[Shot 1] Handheld, fast reframing, hard practical light off stainless "
        "steel. A medium-wide shot establishes {environment_beat}a restaurant line "
        "mid-service, four burners lit, steam crossing the lens, a ticket rail "
        "loaded above the pass. <|caption_start|>TABLE 12 - 2 COVERS - FIRE"
        "<|caption_end|> {character} works the pass, preserving the face, hair, "
        "wardrobe and build established in the reference, slaps the rail and calls "
        "down the line: <d>[English] Two on twelve, fire it now.</d> His lips "
        "close and he snaps the ticket free with two fingers. The camera tracks "
        "right at large amplitude and fast speed past three cooks, one tossing a "
        "pan so the flame climbs above the rim.",
        "[Shot 2] At 00:03.500, the shot cuts to a close shot of a young line cook "
        "with a light, quick soprano (S2) at the flat top, moving fast, who "
        "answers without looking up: <d>[English] Two on twelve, heard.</d> Her "
        "lips close, and she sings along under her breath with a radio on the "
        "shelf behind her: <|lyrics_start|><d>[English] Keep it moving, keep it "
        "hot.</d><|lyrics_end|> Her lips close as she flips two portions in one "
        "motion and the flame flares behind her shoulder.",
        "[Shot 3] At 00:07.500, the camera pushes in fast with large amplitude on "
        "the pass as plates land in a row, hands entering frame from three "
        "directions, a thumb wiping a rim clean. {character} and the cook overlap "
        "at speed: <d>[English] Where is my second plate.</d> <d>[English] Behind "
        "you, behind you.</d> Both sets of lips close as a plate is spun into "
        "position. The camera stays low across the pass so the stainless "
        "surfaces, the loaded ticket rail and the lit burners of the reference "
        "setting remain continuously visible behind the hands, steam crossing "
        "the lens twice without hiding either face.",
        "[Shot 4] At 00:11.500, the shot changes to a low shot as a runner lifts "
        "both plates and turns for the door, the kitchen receding behind him in a "
        "blur of steam, while {character} calls after him already reading the next "
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


def _scene_description(scene: str, image_roles, defs) -> str:
    """A stress scene's `detailed_description`, with the labels this arm wires.

    `{character}` resolves by ROLE, not by ordinal, for the same reason the
    establishing beat does: once roles are declared per socket the character
    can sit anywhere, and an arm whose first socket is a garment would
    otherwise have the busker played by a coat.

    **Refuses rather than under-cites.** Every defined `<Subject N>` has to
    appear in `detailed_description` (ref-en.txt:231); a subject that carries a
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
#: `Video/20260808-stock-vs-vendortokens/`) and kept because it worked, so it
#: is reproduced verbatim rather than rewritten.
#:
#: What makes it a good test, as opposed to a good clip:
#:
#: * **Eight short lines instead of two long ones**, across three shots, so the
#:   clip carries seven speaker changes in fifteen seconds. Lip sync has to be
#:   right repeatedly and at speed; a model that drifts is caught on the next
#:   line rather than at the end.
#: * **The pacing is written, not hoped for.** "answers immediately", "says at
#:   once", and a soundscape line that says the two voices trade "short clipped
#:   lines with almost no gap between them". Remove those and the same lines
#:   come out spaced and unjudgeable.
#: * **Every line is `<d>[English] ...</d>`**, which the prompt guide requires
#:   for all dialogue. The marker is the subject: if the tokenizer is not
#:   emitting 151669/151670 the model hears angle brackets and the word
#:   "English" as prose, and the tell is AUDIBLE -- it speaks them or slurs the
#:   line start. Being able to hear that is why this arm exists.
#: * **Two voices described by register** (measured female S1, lower gravelled
#:   male S2), so a swapped or blended speaker is obvious without a spectrogram.
#:
#: Do not "improve" it without rendering the result beside this one.
DIALOGUE_T2V_PROMPT = """integrated_multimodal_description: [Shot 1] Live-action, cinematic, handheld on 35mm, a medium two-shot frames a woman with dark shoulder-length hair in a charcoal wool coat facing a man with a close-cropped beard in a navy jacket on a concrete stairwell landing, lit hard from a caged bulb overhead. The woman with the low measured voice (S1) says, <d>[English] You said tomorrow.</d> The man with the lower gravelled voice (S2) answers immediately, <d>[English] It moved.</d> She says at once, <d>[English] To when?</d> He says, <d>[English] Tonight.</d> Her jaw tightens and the camera drifts a few degrees with the operator's breathing. [Shot 2] At 00:06.000, the camera cuts to a close-up of the woman against painted cinderblock, the bulb throwing a hard edge down one cheek. She (S1) says, <d>[English] Who else knows?</d> Off screen he (S2) answers immediately, <d>[English] Nobody.</d> She says at once, <d>[English] Keep it that way.</d> She looks past the camera toward the stairs below. [Shot 3] At 00:11.000, the camera cuts to a close-up of the man, rain noise faint through a window well behind him. He (S2) says, <d>[English] Understood.</d> He looks down, adjusts the strap on his shoulder, and holds still until the final frame.

overall_soundscape: Close handheld room tone in a hard concrete stairwell with a long reflective tail on every consonant and a faint electrical hum from the caged bulb overhead. Two speaking voices, a measured adult female voice and a lower adult male voice, trading short clipped lines with almost no gap between them.

non_diegetic_music: N/A
"""

#: The ref2va twin. **Same eight lines, same three shots, same cut times, same
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
DIALOGUE_REF2V_PROMPT = """subject_definitions:
<Subject 1> is the woman shown in <Picture 2>, preserving her facial identity, shoulder-length dark brown hair, and warm features. She wears a charcoal wool coat in the target video. The daylight park background of <Picture 2> is not present in the target video.
<Subject 2> is the elderly man shown in <Picture 1>, preserving his facial identity, white swept-back hair, and deeply lined face. He wears a navy jacket in the target video. The brown studio backdrop of <Picture 1> is not present in the target video.

summary:
[reference generation] In three shots on a concrete stairwell landing, <Subject 1> and <Subject 2> trade eight short clipped lines with almost no gap between them, about a plan that has moved.

retention_analysis:
<Subject 1> (appears in [Shot 1] and [Shot 2]): fully_preserved - retain the same face, shoulder-length dark brown hair and features in every frame she appears in.
<Subject 2> (appears in [Shot 1] and [Shot 3]): fully_preserved - retain the same face, white swept-back hair and deeply lined face in every frame he appears in.

detailed_description:
Photorealistic live-action, 16:9, handheld on 35mm with visible grain, shallow depth of field, lit hard and from above by a single caged bulb on the stairwell ceiling so both faces carry a bright top edge and a deep shadow under the brow and jaw. The stairwell is poured concrete with painted cinderblock walls in a flat institutional green, a steel handrail running down out of frame, and a scuffed landing floor. The air is cold enough to be visible faintly on the breath at the end of the longer lines. The camera stays handheld throughout with small continuous drift from the operator's breathing, never a deliberate move. Each speaker's lips move only on their own line and are closed and still while the other speaks. Both faces stay in focus whenever they are in frame. No other people, no readable text, no signage, no costume changes, no music.

[Shot 1, 00:00.000-00:06.000] A medium two-shot frames <Subject 1> in a charcoal wool coat, collar turned up, facing <Subject 2> in a navy jacket on the stairwell landing, the two of them a little closer than is comfortable. <Subject 1>, with a low measured voice (S1), says: <d>[English] You said tomorrow.</d> <Subject 2>, with a lower gravelled voice (S2), answers immediately: <d>[English] It moved.</d> She says at once: <d>[English] To when?</d> He says: <d>[English] Tonight.</d> Her jaw tightens, she does not step back, and the camera drifts a few degrees with the operator's breathing while the bulb hums overhead.

[Shot 2, 00:06.000-00:11.000] The camera cuts to a close-up of <Subject 1> against the painted cinderblock, the bulb throwing a hard edge down one cheek and leaving the other in shadow, a few strands of hair lit at the crown. She (S1) says: <d>[English] Who else knows?</d> Off screen he (S2) answers immediately: <d>[English] Nobody.</d> She says at once: <d>[English] Keep it that way.</d> She holds his eyeline a moment longer than the line needs, then looks past the camera and down toward the stairs below, her breath just visible.

[Shot 3, 00:11.000-00:15.000] The camera cuts to a close-up of <Subject 2>, the same hard overhead light picking out the lines around his eyes and mouth, rain noise faint through a window well behind him and a wet grey rectangle of light on the wall past his shoulder. He (S2) says: <d>[English] Understood.</d> He looks down, adjusts the strap on his shoulder with one hand, exhales once, and holds still with his lips closed until the final frame.

overall_soundscape:
Close handheld room tone in a hard concrete stairwell with a long reflective tail on every consonant and a faint electrical hum from the caged bulb overhead. Two speaking voices, a measured adult female voice and a lower adult male voice, trading short clipped lines with almost no gap between them.

non_diegetic_music:
N/A
"""


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

    # Guide 3.2's task-type vocabulary. This is NOT cosmetic: it is the only
    # place the prompt states what relationship the references stand in, and
    # every arm shipped `[reference generation]` regardless of role, which
    # collapsed the exact axis these arms exist to vary.
    #
    # 3.2 is explicit that presence does not imply a type -- "if a reference
    # video provides only camera movement, cuts, or rhythm, it normally
    # belongs to `reference generation`" -- so motion and structure stay
    # reference generation and only edit/continue get their own type.
    types = []
    if video and video_role in ("edit", "swap"):
        # A character swap IS a direct modification of the source video, so
        # 3.2 puts it here and not under `reference generation`. Community
        # write-ups of this scenario often stop at a bare `[video editing]`;
        # 3.2 is explicit that reused audible audio adds `audio reuse` too,
        # which the block below supplies.
        types.append("video editing")
    elif video and video_role == "continue":
        types.append("video continuation")
    if images or (video and video_role in ("motion", "structure")):
        types.append("reference generation")
    # 4.2's markers decide the audio type: fully_copy/partially_copy are a
    # reuse of the signal, `reference` is not. 3.2: "when editing a source
    # video, use `audio reuse` as well if its original audio remains audible."
    if video_audio or (audio and audio_role == "copy"):
        types.append("audio reuse")
    if audio and audio_role in ("voice", "music"):
        types.append("audio reference")
    if not types:
        types.append("reference generation")

    return "\n".join([
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
    ])


_NOTE_IMAGE_EDIT = """\
## One frame. This graph is an image editor, and it rests on a patch

H3 renders a single frame if you ask it for one, and at one frame it behaves
like a capable reference-driven image editor. **ComfyUI does not let you ask.**
Its H3 nodes floor `length` at 5 -- the only video family in `comfy_extras`
that floors above 1 (Wan uses `min=1` at all 16 of its length inputs, Hunyuan
at 3, Cosmos at 3). This pack lifts that floor in memory at load
(`single_frame.py`), which means:

**If the shim is disabled, `MiniMaxH3Resolution` refuses the render** and says
why. That refusal is load-bearing and it is ours, not ComfyUI's: **measured
2026-08-15, ComfyUI accepts this graph without the shim and renders five
frames through the single-image VAE, silently.** Its validator enforces a
widget's `min` only on LITERAL values, and this graph wires `length` over a
link from the Resolution node, so core never checks it -- it just clamps 1 up
to 5 at execution. The note here said the opposite until it was tested.
Upstream tracking: Comfy-Org/ComfyUI#15644.

### What is different from every other graph here

| | this graph | the video graphs |
|---|---|---|
| length | **1** | 124-362 |
| VAE | **single-image H3 VAE** | `minimax_h3_video_vae_fp16` |
| audio | no decoder at all | decoded and muxed |
| output | `SaveImage` | `VHS_VideoCombine` |

**The VAE is the half that is easy to get wrong.** It is the same checkpoint
with a decoder retrained to reconstruct one image from a single temporal
latent -- verified from the safetensors, not its README: 121 of 562 tensors are
byte-identical to the stock video VAE, being all 116 encoder tensors,
`quant_conv` and the latent statistics, while the 441 that differ are the
decoder plus `post_quant_conv`. **The encoder is frozen, so the latent space is
identical and this is purely a decoder swap.** Its own README warns it
regresses multi-frame reconstruction, with patch-grid ghosting and cross-frame
mixing. Never put it in a video graph.

**Measured here 2026-08-15, with ground truth, because "you need the special
VAE" was worth checking rather than repeating.** Round-tripping this graph's
own reference image (encode then decode at T=1, so the source IS the target):

| decoder | PSNR | SSIM | mean abs error |
|---|---|---|---|
| single-image VAE | **37.27 dB** | 0.947 | 1.95/255 |
| stock video VAE fp16 | 22.04 dB | 0.821 | 14.72/255 |

15.2 dB. So the swap is not a preference. **Core decodes T=1 with either** --
the video VAE does not fail, it just returns a harsher, colour-shifted image,
which is the trap: it looks like a working render.

And the artifact the community reported is real and reproduces: decoding a
5-frame latent with this VAE and keeping frame 0 leaves gradient energy
aligned to the patch grid at 1.46x (16px) and 1.50x (32px) the off-grid
average, against 1.03-1.22x for every other combination tried.

Core was already ready for this: `comfy/ldm/minimax/vae.py` has an explicit
`t == 1` branch, and it keeps the LAST of the 4 frames one latent decodes to --
which is exactly the `h3_t1_output_slice: 3` the VAE's metadata declares. The
node floor was the only thing in the way.

### Without the shim, the fallback is worse and it is not the same thing

Render `length=5` with the stock video VAE and keep frame 0. It works, and the
community reports it comes out soft. Note what that fallback actually is: the
DiT denoises 2 latent temporal steps instead of 1, so it costs about twice the
video rows, and the decode is a video decode you then throw 4 frames of away.

### Where this graph deliberately differs from the community workflow

It follows the r/StableDiffusion single-image-edit write-up (2026-08-14), and
departs from it in four places, each on purpose:

- **Canvas 768x1152, not 1024x1536.** Theirs is 1.57 MP, which is 52% over
  H3's 768*1344 area cap and outside the trained family. Ours is the in-family
  2:3. Theirs is not wrong -- it renders, and bigger may well look better --
  but it is a different question, and `MiniMaxH3Resolution`'s `custom` option
  reaches it and says which side of the family you are on.
- **sage fp16, not Comfy Kitchen attention.** Theirs carried CK over from a
  video workflow; the author re-ran without it and reported quality slightly
  improved and speed unchanged.
- **Base ref2va, no turbo LoRA.** Theirs stacks a hybrid fl2va/ref2va
  checkpoint plus a turbo LoRA plus a detail LoRA. Each is plausible and each
  is a variable; this is the baseline they should be measured against.
- **One reference, not several.** The question an edit model has to answer is
  whether identity survives the change.

### The cost lever here is NOT the canvas

At one frame the video segment is a single latent step, so the shape of the
sequence is nothing like a video render. Measured by Preflight on this graph:

```
sequence length 9,240      text        4,276   46.3%
768x1152, trained family   references  4,096   44.3%
864 video tokens/frame     video         864    9.4%
                           audio           4    0.0%
```

**The video is 9% of it, and the reference is nearly all the rest.** Read that
`text` row carefully: the prompt is under 200 tokens of it. The other ~4,100
are the reference image again, as Qwen vision tokens -- **every reference is
paid for twice**, once in the text segment and once as latent rows, and both
ride every sampling step. Measured across a 1/2/3/4/6 ladder on 2026-08-15, the
text half scales with reference COUNT and lands 75-160 rows *above* the
reference half at every rung (see `docs/h3_references.md`).

So changing the canvas moves almost nothing here -- 1:1 saves 3%, 16:9 costs
2% -- where in a 124-frame render it is the single largest lever. What costs is
the references, doubled. Nine of them at this sizing is ~94k rows and **OOMs a
24 GB 4090**, which is more than the 124-frame video graph asks for.

**The numbers above are the `allow_upscale=True` shape, which this graph no
longer ships.** They are kept because they are what a reference costs when the
fit node takes it to 2048, and that is still one `ref_upscale=True` away.

### `allow_upscale` is off here, and it was the whole cost

The fit node's upscaling is the single largest lever on this path. Measured at
one seed on the two-reference scene, then confirmed here 2026-08-16 on the
shipped graph:

| sizing | ref rows | secs |
|---|---:|---:|
| `max` + fit upscale (what this used to ship) | 8,192 | 84 |
| `max`, no fit upscale (**ships now**) | 2,048 | 18 |
| `match` | 1,682 | 16 |

4.9x the rows and 5.2x the wall clock, and at 1:1 on the face all three held
the same identity, glasses, hair and features. `docs/open_experiments.md` #16e
has the caveat that matters: that comparison is one subject at one seed, and it
is why the VIDEO graphs have not moved.

`ref_image_size` stays `max` (2048 short edge) and is still a **no-op** for
every reference in `h3_refs/`, but for a different reason than before, and the
old one is now wrong. It used to be a no-op because the fit node had already
reached 2048, so core's `min(1.0, 2048 / short_edge)` was 1.0. Now the fit node
leaves the source alone and a sub-2048 reference hits `min(1.0, >1.0)` = 1.0
instead. Same outcome, different mechanism -- and above 2048 the two diverge,
so it is not redundant.

For scale, measured by `bench/preflight_graph.py` rather than estimated -- an
earlier version of this paragraph said "~5,200" from arithmetic and was 58%
high:

```
h3_image_edit          3,282     1 reference
h3_image_recolor       3,304     1
h3_image_sheet         3,260     1
h3_image_style         5,386     2
h3_image_composite     5,419     2
h3_image_multiperson   7,520     3
```

Against ~82,686 for the 124-frame reference video graph, which is why these
render in seconds."""


# --------------------------------------------------------------------------
# The single-frame image gen/edit prompts
# --------------------------------------------------------------------------
#
# **This reverses a decision, so read why before reverting it.** Until
# 2026-08-16 there was one image prompt, `_image_edit_prompt`, and its
# docstring argued at length that the guide format *cannot* apply to a still:
# two of its six sections are audio, and `detailed_description` is specified as
# `[Shot 1]` with camera movement and shot timing, none of which a one-frame
# render has. So it shipped a plain paragraph in the form the community's
# first write-up used.
#
# What changed is evidence, not taste. The author of that write-up published a
# second set on 2026-08-15 (`internal/refs/`), and between the two posts they
# switched formats: post 1 is flat `Task: Reference-guided generation. ...`
# prose, post 2 is the guide's structure with the two audio sections dropped.
# The move is in the direction the old docstring argued against, by someone
# who had rendered a couple of thousand images on this path.
#
# That is a reason to test, not a reason to believe. **Neither post is a
# controlled comparison** -- the scenes differ, the references differ, and
# nothing was held fixed -- so what we have is a practitioner's revealed
# preference, which is the same grade of evidence as the Custom-GPT kit in
# `internal/PROMPTING.md` section 4.2. Hence the ladder below rather than a
# rewrite.
#
# The half of the old argument that survives intact: the audio sections
# describe something a single-frame graph structurally cannot produce (it has
# no `VAEDecodeAudio` at all). That is why `sections` is the default and `av`
# is the arm, and not the other way round.

# The three formats, as a ladder. Each rung removes exactly one thing, so a
# difference between two arms has one candidate cause.
#
#   av        all six guide sections, audio ones present and "N/A"
#   sections  the four visual sections            <- av minus the audio pair
#   flat      one paragraph, no headers, no [Shot 1]
#                                                 <- sections minus scaffolding
#
# `flat` drops the shot marker as well as the headers, deliberately: it is the
# community's post-1 form and this repo's own previous shipped form, and both
# are unscaffolded prose. So B->C is "all remaining structure", not "headers
# only". Stated because a two-thing rung is the kind of detail that gets
# forgotten and then mis-attributed.
#
# **`flat` keeps `<Subject N>` even though the community's post-1 prompts do
# not**, and that is a deliberate departure from reproducing their form. The
# subject labels are the only place the reference roles are stated, so
# dropping them would change what the arm SAYS as well as how it is laid out,
# and the comparison would no longer be about format. If the structured arms
# win, whether the subject indirection specifically is what did it is a
# separate follow-up and a separate arm.
IMAGE_FORMATS = ("av", "sections", "flat")

# What each reference DOES, per scene. The whole point of the exercise: a
# reference the prompt never assigns a job to still costs its rows on every
# sampling step, and the model has to guess what it was for.
#
# **Content is written ONCE per scene and rendered into all three formats.**
# Hand-writing a flat variant would have let the arms differ in wording as
# well as in structure, which would measure the writing and report it as the
# format. Same sentences, different scaffolding, or the ladder means nothing.
#
# Every scene names an `h3_refs/` asset from `internal/reference_library.md`,
# so the subject of a result is documented rather than being whatever was in
# the input root that day. `face_elderly_man_suit_1024x1024.png` is
# byte-identical to the `1-man.png` this path used before (md5 f277a530...),
# so the camera scene is the same render it always was, under the name that
# says what it is.
#
# **Scenes are drawn from the two r/StableDiffusion write-ups**, chosen so each
# exercises a different retention marker rather than a different subject:
# fully_preserved, partially_preserved and attribute_transfer all appear, and
# `style` is the one where getting the roles wrong is visible at a glance --
# a style reference that leaks its own content produces a cottage.
_IMAGE_SCENES: dict[str, dict] = {
    # The scene that has to stay honest about what it is testing. Its first
    # version asked to age the subject to 60 against a reference of a man well
    # past 70: it rendered, it looked like a working edit, and it demonstrated
    # only that the pipeline runs. A prompt the input already satisfies cannot
    # fail. A camera move cannot be a no-op on a fixed photograph, and it is
    # the capability worth showing -- rotating the camera while keeping the
    # room and the person consistent is what image edit models are worst at
    # and what a video model is structurally good at.
    "camera": dict(
        refs=("h3_refs/face_elderly_man_suit_1024x1024.png",),
        subjects=[
            "<Subject 1> is the man in <Picture 1>, with his own facial "
            "structure, eyes, nose, mouth, ears, skin tone and texture, white "
            "hair and hairline, dark suit, white shirt and navy tie.",
        ],
        summary="Re-photograph <Subject 1> from a camera moved to his left "
                "and slightly down, keeping the studio, the wardrobe and the "
                "key light of <Picture 1> unchanged",
        retention=[
            "<Subject 1>: partially_preserved - identity, age, wardrobe, "
            "background and lighting are retained; only the camera position "
            "and the resulting occlusions change.",
        ],
        style="One realistic portrait photograph in the same photographic "
              "style as <Picture 1>.",
        body="The camera sits about 45 degrees to <Subject 1>'s left and "
             "slightly below its original height, so he is seen in "
             "three-quarter view rather than facing the lens. <Subject 1> turns "
             "his head to follow the camera and looks directly into it, while "
             "his shoulders stay squared to his original facing, so the turn "
             "reads in the neck and head and not in the torso. The newly "
             "visible side of his face and head is consistent with the "
             "original view. The plain brown studio background and the soft "
             "directional key light falling from the same side are unchanged.",
    ),

    # The character swap, on the path where it costs seconds instead of
    # minutes. `h3_ref_video_swap` asks the same thing of a video plate and
    # one identity; this asks it of a still plate and TWO, which is the case
    # the video arms do not cover and the one where the reported failure
    # lives -- the model blending two identities, or putting one person's
    # features on the other.
    #
    # **Every attribute below was read off the plate at full resolution**,
    # not inferred from the thumbnail. The first draft of this scene had the
    # woman sitting with her knees drawn up and the man's floral jacket
    # draped over her legs; she is lying prone on her forearms with her boots
    # in the air, and the jacket is his. A prompt asserting a pose the plate
    # does not hold asks the model to reconcile the two, which is the
    # generic-template failure this file records above.
    #
    # **Chosen so a failure cannot pass for a success.** Both people in the
    # plate are young with dark hair; the two identities are a freckled
    # middle-aged redhead and a curly-haired man in black-rimmed glasses. If
    # the swap does not happen, or happens on the wrong person, it is visible
    # at a glance rather than a judgement about likeness. A plate whose
    # occupants resembled the replacements would render something plausible
    # and demonstrate nothing.
    "swap": dict(
        refs=("h3_refs/scene_loft_couch_duo_2752x1536.png",
              "h3_refs/face_freckled_woman_redhair_1024x1024.png",
              "h3_refs/face_young_man_glasses_1024x1024.png"),
        subjects=[
            "<Subject 1> is the woman at camera-left in <Picture 1>, with her "
            "identity replaced: her face, skin, freckling, hair colour and "
            "length, and apparent age come exclusively from <Picture 2>. Her "
            "pose lying prone along the couch propped on her forearms with "
            "her knees bent and her boots raised behind her, her dark hair "
            "gathered up off her neck, her black sleeveless top and dark "
            "trousers, and her position and scale in frame are those of "
            "<Picture 1>.",
            "<Subject 2> is the man at camera-right in <Picture 1>, with his "
            "identity replaced: his face, skin, hair and black-rimmed glasses "
            "come exclusively from <Picture 3>. His upright seated posture, "
            "his white shirt and gold-and-black floral jacket, his eyeline off "
            "camera-left, and his position and scale in frame are those of "
            "<Picture 1>.",
            "<Subject 3> is the loft interior of <Picture 1>: the raw concrete "
            "wall, the daylight window at camera-left, the black leather "
            "couch, the glass table with the yellow book and the red "
            "telephone on it, and the cool desaturated grade.",
            "<Picture 2> and <Picture 3> supply facial identity only. Neither "
            "supplies lighting, exposure, colour grade, background, pose, "
            "clothing, framing or composition.",
        ],
        summary="Replace the identities of the two people in <Picture 1> with "
                "those of <Picture 2> and <Picture 3>, keeping the loft, the "
                "couch, both poses, both outfits, the framing and the light "
                "exactly as they are",
        retention=[
            "<Subject 1>: attribute_transfer - the facial identity of "
            "<Picture 2> is transferred onto the woman's pose, wardrobe and "
            "position from <Picture 1>.",
            "<Subject 2>: attribute_transfer - the facial identity of "
            "<Picture 3> is transferred onto the man's pose, wardrobe and "
            "position from <Picture 1>.",
            "<Subject 3>: fully_preserved - the loft, couch, table, objects, "
            "window light and colour grade of <Picture 1> are unchanged.",
        ],
        style="One realistic photograph in the same photographic style, grain "
              "and colour grade as <Picture 1>.",
        body="<Subject 1> and <Subject 2> occupy exactly the positions they "
             "hold in <Picture 1>, at the same scale and in the same framing: "
             "she at camera-left, lying prone along the couch on her forearms "
             "with her boots raised behind her; he at camera-right, sitting "
             "upright in the gold-and-black floral jacket and looking off "
             "camera-left. "
             "Only the two faces change. The daylight from camera-left falls "
             "on both new faces from the same direction and at the same "
             "softness as it falls on the originals, and neither new face "
             "brings its own lighting, background or crop into the frame. No "
             "feature of <Picture 2> appears on <Subject 2> and no feature of "
             "<Picture 3> appears on <Subject 1>. <Subject 3> is unchanged in "
             "every detail, the yellow book and the red telephone on the glass "
             "table included. Exactly two people appear anywhere in the frame.",
    ),

    # Two references with opposite jobs, and the one scene where a role
    # mistake is unmissable: if <Picture 2> is read as content rather than as
    # technique, a cottage and a woodland arrive with the graphite.
    "style": dict(
        refs=("h3_refs/face_freckled_woman_redhair_1024x1024.png",
              "h3_refs/style_pencil_cottage_1024x1024.png"),
        subjects=[
            "<Subject 1> is the adult woman in <Picture 1>, with her own "
            "facial geometry, expression, gaze, freckling, red hair and head "
            "angle.",
            "<Subject 2> is the graphite drawing technique in <Picture 2>: its "
            "pencil contours, hatching, tonal modelling, erased highlights and "
            "visible paper. <Picture 2> supplies no subject, no scene and no "
            "composition.",
        ],
        summary="Convert <Subject 1> into one finished graphite portrait, "
                "transferring only the drawing medium of <Subject 2>",
        retention=[
            "<Subject 1>: fully_preserved - identity, facial geometry, "
            "expression, gaze, hairstyle, head angle, crop and the lighting "
            "relationships are retained.",
            "<Subject 2>: attribute_transfer - its graphite handling is "
            "applied to <Subject 1> without copying its cottage, its woodland "
            "or its composition.",
        ],
        style="One monochrome graphite drawing on off-white paper.",
        body="<Subject 1> is rendered in the technique of <Subject 2>: precise "
             "pencil contours, varied pressure, fine parallel and cross "
             "hatching, soft tonal modelling, erased highlights and visible "
             "paper tooth. <Subject 1>'s face and expression are preserved "
             "while photographic microtexture becomes drawn value and "
             "mark-making. Every region is converted to the medium of "
             "<Subject 2> consistently, including hair, skin, clothing and "
             "background; no area stays photographic or coloured, and no "
             "cottage, woodland or other content from <Subject 2> appears. "
             "Exactly one adult, and no added person, text, signature or "
             "decorative frame.",
    ),

    # Identity against a whole new environment. The failure this scene is
    # written to expose is the cutout: correct pixels, wrong light, no contact
    # shadow, and the person visibly pasted onto a plate.
    "composite": dict(
        refs=("h3_refs/face_young_man_glasses_1024x1024.png",
              "h3_refs/scene_alpine_lake_meadow_1024x1024.png"),
        subjects=[
            "<Subject 1> is the young man in <Picture 1>, with his own face, "
            "curly hair, black-rimmed glasses, build and clothing.",
            "<Subject 2> is the outdoor environment in <Picture 2>: its "
            "meadow, lake, mountains, palette, daylight direction and depth. "
            "<Picture 2> supplies no person.",
        ],
        summary="Place <Subject 1> inside <Subject 2> as one photograph taken "
                "in that location",
        retention=[
            "<Subject 1>: partially_preserved - face, hair, glasses, build and "
            "clothing are retained; the studio background, its flat "
            "illumination and the original framing are not.",
            "<Subject 2>: fully_preserved - the meadow, lake, mountains, "
            "palette and daylight are the complete replacement environment.",
        ],
        style="One realistic outdoor photograph, single exposure.",
        body="<Subject 1> stands in the foreground meadow of <Subject 2>, framed "
             "from the knees up and turned slightly away from the lake. His "
             "studio background is gone entirely. <Subject 1> is relit to "
             "belong to <Subject 2>: its daylight direction produces coherent "
             "highlights and shaded planes across his face, glasses, hair and "
             "clothing, the flat studio illumination does not survive, and cool "
             "reflected light from the water reaches his shaded side. His feet "
             "meet the ground of <Subject 2> with a dark contact patch and one "
             "connected cast shadow running in the same direction and softness "
             "as the shadows already in the meadow. Perspective, scale, colour "
             "temperature and depth of field agree with <Subject 2>, so the "
             "result reads as one camera exposure rather than a cutout. "
             "Exactly one person, and no halo, pasted edge or floating feet.",
    ),

    # Three references, two of them people. Identity separation is the
    # question, and it is the one thing the cost arithmetic cannot predict:
    # 2026-08-16 measured four and six references composing cleanly, so what
    # this scene asks is whether the prompt can still say WHICH person is
    # which once there are two faces in front of it.
    "multiperson": dict(
        refs=("h3_refs/face_young_man_glasses_1024x1024.png",
              "h3_refs/face_freckled_woman_redhair_1024x1024.png",
              "h3_refs/scene_officers_corridor_1376x768.jpeg"),
        subjects=[
            "<Subject 1> is the young man in <Picture 1>, with his own face, "
            "curly hair, black-rimmed glasses, build and clothing.",
            "<Subject 2> is the adult woman in <Picture 2>, with her own face, "
            "freckling, red hair, build and clothing.",
            "<Subject 3> is the green-lit marble corridor in <Picture 3>: its "
            "architecture, palette, lighting and depth. <Picture 3> supplies "
            "no person.",
        ],
        summary="Place <Subject 1> and <Subject 2> together in <Subject 3> as "
                "one photograph of two people in conversation",
        retention=[
            "<Subject 1>: partially_preserved - face, hair, glasses, build and "
            "clothing are retained; pose, framing and lighting change.",
            "<Subject 2>: partially_preserved - face, freckling, hair, build "
            "and clothing are retained; pose, framing and lighting change.",
            "<Subject 3>: fully_preserved - the corridor is the complete "
            "environment, with its own architecture, palette and green light.",
        ],
        style="One realistic photograph, medium-wide, single exposure.",
        body="The two adults stand an arm's length apart in the middle of the "
             "corridor, angled toward each other. <Subject 1> is camera-left "
             "with one hand at his side and his head turned toward her; "
             "<Subject 2> is camera-right, speaking, one hand raised at chest "
             "height. Each keeps their own face, hair, build and clothing with "
             "no blending between them and no feature of one appearing on the "
             "other. Their eyelines meet, their scale agrees with the corridor, "
             "and both sets of feet meet the floor with contact shadows in the "
             "same direction as the architecture's own. The green key light of "
             "<Subject 3> falls across both of them. Exactly two people appear "
             "anywhere in the frame, and the corridor behind them stays empty.",
    ),

    # The strictest retention case in the set: everything is held except two
    # named attributes. It is here because "change only X" is where an edit
    # model usually drifts wardrobe, crop or expression while nobody is
    # looking at them, and because the reference cannot already satisfy it.
    "recolor": dict(
        refs=("h3_refs/face_freckled_woman_redhair_1024x1024.png",),
        subjects=[
            "<Subject 1> is the adult woman and the complete portrait image in "
            "<Picture 1>, including her clothing, the background, the crop and "
            "the lighting.",
        ],
        summary="Make one selective colour edit to <Subject 1>: her visible "
                "skin becomes sapphire blue and her hair becomes silver-white, "
                "in the same portrait photograph",
        retention=[
            "<Subject 1>: partially_preserved - skin colour and hair colour "
            "change; identity, facial geometry, age, expression, gaze, pose, "
            "crop, clothing, background, lighting, camera angle and depth of "
            "field are all retained.",
        ],
        style="One photorealistic portrait photograph.",
        body="Exactly two colour attributes of <Subject 1> change. All visible "
             "skin becomes a rich, unmistakable sapphire blue while keeping its "
             "pores, freckling pattern, shading, highlights and tonal depth. "
             "All hair becomes luminous silver-white while keeping the exact "
             "hairline, strand detail, shape, volume and shadows. The face of "
             "<Subject 1> is the same face: the same eyes, the same "
             "expression, the same gaze, the same head angle. Everything else "
             "in <Subject 1> is untouched -- clothing keeps its colour, "
             "material, folds, highlights and shadows, and the background is "
             "unchanged. No makeup is added, no facial feature is altered, and "
             "no object is changed. Exactly one adult, and no text or border.",
    ),

    # Geometric consistency from a single view, which is the thing a video
    # model should be structurally good at and an image editor is not. Read it
    # against `camera`: same capability, one view against three.
    "sheet": dict(
        refs=("h3_refs/face_young_man_glasses_1024x1024.png",),
        subjects=[
            "<Subject 1> is the young man in <Picture 1>, with his own face, "
            "curly hair, black-rimmed glasses, build, clothing and footwear.",
        ],
        summary="Present <Subject 1> as one character sheet of three "
                "consistent views",
        retention=[
            "<Subject 1>: fully_preserved - face, hair, glasses, build, "
            "clothing and footwear are identical in all three views; only the "
            "viewing angle differs.",
        ],
        style="One clean photographic character sheet on a seamless "
              "light-grey studio ground.",
        body="Three full-body views of <Subject 1> stand side by side on one "
             "canvas: front, side and rear, in that order left to right, at the "
             "same height and the same distance from the camera. Every view "
             "carries the identical face, hair, glasses, body and clothing of "
             "<Subject 1>, and the rear view's hair, collar and footwear follow "
             "from the front view rather than being invented freely. "
             "<Subject 1> holds a neutral relaxed stance with arms clear of the "
             "torso and both feet visible in each view. Even studio lighting "
             "falls the same way on all three. No captions, labels, borders or "
             "panel gutters.",
    ),
}


# Guide section 4.1's visual markers, in the English the flat arm uses. Audio
# markers are absent because a single-frame graph has no audio layer to give
# one to.
_MARKER_PROSE = {
    "fully_preserved": "is fully preserved",
    "partially_preserved": "is partially preserved",
    "attribute_transfer": "supplies an attribute transfer",
    "weak_reference": "is a weak reference",
}


def _marker_to_prose(line: str) -> str:
    """`<Subject 1>: fully_preserved - x` -> `<Subject 1> is fully preserved: x`.

    Raises rather than passing an unknown marker through: a marker this does
    not recognise is either a typo or a fifth marker, and both mean the flat
    arm would silently carry different text from its twin -- which is the one
    thing that would make the comparison meaningless.
    """
    m = re.match(r"(<Subject \d+>): (\w+) - (.*)$", line, re.S)
    if not m or m.group(2) not in _MARKER_PROSE:
        raise ValueError(f"retention line is not `<Subject N>: <marker> - ...` "
                         f"with a known marker: {line!r}")
    return f"{m.group(1)} {_MARKER_PROSE[m.group(2)]}: {m.group(3)}"


def _image_prompt(scene: str = "camera", fmt: str = "sections") -> str:
    """A single-frame image gen/edit prompt, in one of three formats.

    `scene` selects the content from `_IMAGE_SCENES`; `fmt` selects the
    scaffolding from `IMAGE_FORMATS`. Content and format are separate on
    purpose -- see the ladder note above.

    What every format guarantees, because these are the parts that are not
    stylistic:

    - **Every `<Picture N>` the graph wires gets a job, and only jobs the
      graph can honour.** `check_ref_prompt_labels` fails the build otherwise,
      in any format, and it is not waived for image graphs: naming a reference
      that is not wired is wrong however the prompt is laid out.
    - **A reference that supplies technique says what it does NOT supply.**
      The official guide never writes a negative clause -- every relationship
      there is stated as what a reference provides -- so this comes from
      general prompting research and from the community write-ups, where the
      reported failure is a style reference dragging its own content along.
      Untested here, like the same technique in `_ref_prompt`'s swap arm.
    - **Retention markers stay inside the guide's visual set**
      (fully_preserved / partially_preserved / attribute_transfer /
      weak_reference). `check_prompt_guide_conformance` enforces that on image
      graphs unwaived, because a marker is vocabulary rather than structure.
    """
    if fmt not in IMAGE_FORMATS:
        raise ValueError(f"unknown image prompt format {fmt!r}; "
                         f"expected one of {IMAGE_FORMATS}")
    if scene not in _IMAGE_SCENES:
        raise ValueError(f"unknown image scene {scene!r}; "
                         f"expected one of {tuple(_IMAGE_SCENES)}")
    s = _IMAGE_SCENES[scene]

    # `reference generation` and nothing else, from guide section 3.2. The
    # other five types describe relationships a still frame cannot stand in:
    # there is no source video to edit or continue, no audio to reuse or
    # reference, and a reference here is guidance rather than a frame anchor
    # of the target, which is what `keyframe completion` means.
    summary = "[reference generation] " + s["summary"] + "."

    if fmt == "flat":
        # One paragraph, no headers, no shot marker. The community's post-1
        # form and this repo's previous shipped form. Same sentences as the
        # structured arms, so the only variable is the scaffolding.
        #
        # **The retention markers become English here, and that is on
        # purpose.** Leaving `attribute_transfer - ...` sitting mid-paragraph
        # would produce a form nobody writes, and an arm nobody would write is
        # a strawman: if it rendered worse, "the structure wins" and "loose
        # vocabulary tokens are noise" would be indistinguishable. So this rung
        # removes the guide's formal apparatus as a UNIT -- headers, shot
        # marker and marker vocabulary -- which is the thing actually in
        # question, and keeps every clause's content word for word.
        return " ".join([
            "Task: reference-guided single-image edit.",
            *s["subjects"], *[_marker_to_prose(r) for r in s["retention"]],
            s["style"], s["body"],
        ])

    out = [
        "subject_definitions:", *s["subjects"], "",
        "summary:", summary, "",
        "retention_analysis:", *s["retention"], "",
        # Guide section 5.3 wants the style stated on its own line BEFORE
        # [Shot 1] on the reference path, not inside it -- the opposite of the
        # t2v rule, and the case `check_prompt_guide_conformance` reads.
        "detailed_description:", s["style"], "[Shot 1] " + s["body"],
    ]
    if fmt == "av":
        # The arm. "N/A" is the guide's own value for an absent layer, so this
        # is the most conformant thing a graph with no audio decoder can say
        # -- which is exactly the question: does carrying the sections at all
        # cost anything on a still?
        out += ["", "overall_soundscape:", "N/A",
                "", "non_diegetic_music:", "N/A"]
    return "\n".join(out)


def _note_image_scene(what: str, watch: str) -> str:
    """Note for a canonical image graph: what it asks for, what to look at."""
    return f"""\
## Single-frame image edit: {what}

One frame, so this is an image editor rather than a video render. The path,
the VAE and the shim it rests on are documented once in `h3_image_edit.json`
and in `docs/h3_image_editing.md`; this note is only about this scene.

**References, and their jobs.** Every `<Picture N>` this graph wires is given
an explicit role in `subject_definitions`, and the ones that supply technique
rather than content also say what they do *not* supply. An unassigned
reference still costs its rows on every sampling step and the model has to
guess what it was for.

**What to look at:** {watch}

**The prompt format is the four visual guide sections**, not the six. The two
audio ones describe a track this graph has no decoder for. Whether that is the
right call is what `h3_image_probe_format_av.json` exists to answer -- render
that and this one's twin scene together before assuming either way.
"""


def _image_graphs() -> tuple:
    """The `GRAPHS` rows for the single-frame image path.

    Kept as a function rather than inlined so the scene table stays the one
    place a scene is described: a row here is a filename, a scene name and a
    format, and everything about what the render CONTAINS lives in
    `_IMAGE_SCENES`.
    """
    def scene(fname, label, scene_name, note, *, fmt="sections", extra=None):
        s = _IMAGE_SCENES[scene_name]
        # `extra` OVERRIDES rather than adds. It was a merge until 2026-08-22,
        # which meant a scene could not restate a key IMAGE_EDIT_BUDGET
        # already set -- `dict(**a, **b)` raises on a collision, so the only
        # way to change the canvas for one scene was to change it for all of
        # them. No caller relied on the old behaviour; a collision could not
        # have shipped, it would have crashed the build.
        spec = dict(single_frame=True, length=1, ref_images=s["refs"],
                    **IMAGE_EDIT_BUDGET,
                    out_prefix=f"Image/{fname.removesuffix('.json')}",
                    variant_note=note)
        spec.update(extra or {})
        return (fname, label, "r2v", _image_prompt(scene_name, fmt), spec,
                f"{len(s['refs'])} reference image(s) -> ONE image: "
                f"{scene_name}, {fmt} prompt")

    return (
        # The canonical graph, and the one carrying the long note about the
        # path itself. Its scene is a camera move because that is the one
        # thing this reference cannot already satisfy -- the version before it
        # asked to age a man well past 70 to 60, which rendered, looked like a
        # working edit, and proved only that the pipeline runs.
        scene("h3_image_edit.json", "r2i", "camera", _NOTE_IMAGE_EDIT),

        scene("h3_image_style.json", "r2i-style", "style",
              _note_image_scene(
                  "a style reference that must not bring its own content",
                  "whether the drawing technique of <Picture 2> arrives "
                  "WITHOUT its cottage and woodland. That is the whole test: "
                  "a style reference read as content is the most common "
                  "multi-reference failure, and here it is unmissable. Then "
                  "whether the likeness in <Picture 1> survives the medium "
                  "change, and whether any region stays photographic.")),

        scene("h3_image_composite.json", "r2i-composite", "composite",
              _note_image_scene(
                  "one identity relit into a different environment",
                  "the contact shadow and the light direction, before the "
                  "face. A composite fails as a CUTOUT long before it fails "
                  "as a likeness: correct pixels, studio lighting still on "
                  "them, no shadow where the feet meet the ground. The prompt "
                  "asks for the studio illumination not to survive, which is "
                  "a harder request than it reads as.")),

        # The image-path twin of h3_ref_video_swap, and the only graph here
        # that swaps TWO identities at once. Render it before spending a
        # video arm on the two-identity case: same question, seconds instead
        # of minutes.
        scene("h3_image_swap.json", "r2i-swap", "swap",
              # The ONLY image scene whose canvas is not the shared 2:3
              # portrait, and the reason is structural rather than taste: the
              # prompt promises the plate's framing survives, and a portrait
              # output cannot hold a 16:9 plate's framing however the model
              # tries. `ASPECTS["16x9"]` is 1.75 against the plate's 1.79 --
              # close, not equal, and the small recompose that implies is a
              # known cost of the scene rather than a swap failure. Read the
              # edges before reading the faces.
              extra=dict(zip(("width", "height"), ASPECTS["16x9"])),
              note=_note_image_scene(
                  "two identities replaced inside one plate",
                  "whether each face lands on the RIGHT person. A swap that "
                  "happens on one of the two, or blends the pair, is the "
                  "reported failure and it is unmissable here -- both "
                  "originals are young and dark-haired and neither "
                  "replacement is. Then whether the plate survives: the "
                  "couch, the window light, the two outfits and the objects "
                  "on the glass table have to come through untouched, since "
                  "a swap that also redecorates the room has not done the "
                  "job asked of it.")),

        scene("h3_image_multiperson.json", "r2i-multiperson", "multiperson",
              _note_image_scene(
                  "two identities in one frame, plus a place",
                  "whether the two faces stay two people. 2026-08-16 measured "
                  "four and six references composing cleanly on this path, so "
                  "the cost side is answered and the open question is "
                  "attribution -- does the prompt still control WHICH person "
                  "is which once there are two of them. Watch for features of "
                  "one appearing on the other, and for a third person.")),

        scene("h3_image_recolor.json", "r2i-recolor", "recolor",
              _note_image_scene(
                  "changing exactly two attributes and nothing else",
                  "everything that was NOT asked to change. The named edit "
                  "(skin, hair) is the easy half; the test is whether the "
                  "crop, expression, gaze, clothing colour, folds and "
                  "background all survive it. Edit models drift wardrobe "
                  "while nobody is looking at the wardrobe.")),

        scene("h3_image_sheet.json", "r2i-sheet", "sheet",
              _note_image_scene(
                  "three consistent views from one",
                  "the rear view, which is the only one with no source "
                  "pixels behind it. Hair, collar and footwear there have to "
                  "FOLLOW from the front view rather than be invented, and "
                  "that is the geometric consistency a video model should be "
                  "structurally better at than an image editor.")),

        # --- the format ladder --------------------------------------------
        #
        # Both arms are the `style` scene, so their twin is
        # `h3_image_style.json` and the ONLY difference is the scaffolding --
        # the sentences are generated from one scene entry for all three. See
        # the ladder note above `IMAGE_FORMATS`.
        #
        # `style` rather than `camera` because it is the scene where the
        # reference roles carry the most weight: one reference supplies
        # identity, the other supplies technique and is explicitly told it
        # supplies nothing else. If structure helps anywhere, it helps here.
        scene("h3_image_probe_format_av.json", "r2i-fmt-av", "style",
              _probe_note(
                  "whether the two audio sections cost anything on a still",
                  "h3_image_style.json",
                  "all six guide sections instead of four, with "
                  "`overall_soundscape` and `non_diegetic_music` present and "
                  "set to the guide's own `N/A`. Same scene, same references, "
                  "same seed.",
                  "the image, against its twin. There is no audio to judge -- "
                  "this graph has no `VAEDecodeAudio` at all -- so the "
                  "question is purely whether carrying two more section "
                  "headers changes what gets drawn.",
                  "no visible difference, which is the useful outcome: it "
                  "would mean the four-section default is free of risk and "
                  "the shorter prompt is simply cheaper. A visible difference "
                  "is the more interesting result and would mean conditioning "
                  "on section headers reaches the image, which nothing here "
                  "has ever shown.",
                  held="same scene, same references, same canvas"),
              fmt="av"),

        scene("h3_image_probe_format_flat.json", "r2i-fmt-flat", "style",
              _probe_note(
                  "whether the guide structure earns its tokens on a still",
                  "h3_image_style.json",
                  "one unbroken paragraph: no section headers, no `[Shot 1]`, "
                  "the same sentences in the same order. This is the form the "
                  "community's first write-up used and the form this repo "
                  "shipped until 2026-08-16.",
                  "whether the roles still bind. The structured twin states "
                  "`attribute_transfer` on the style reference in its own "
                  "section; here the same clause is mid-paragraph. If "
                  "structure matters, this is where the cottage shows up.",
                  "genuinely open, and it is the reason this arm exists. The "
                  "author of the write-up switched from this form to the "
                  "structured one between their two posts, which is a "
                  "practitioner's revealed preference and not a controlled "
                  "comparison -- neither post held the scene or the "
                  "references fixed. This pair does.",
                  held="same scene, same references, same canvas"),
              fmt="flat"),
    )


_NOTE_TURBO_PACK = """\
## A different turbo LoRA, and a different loader on purpose

Read against `h3_probe_ref2v_turbo.json`. Same task, same references, same
seed. The variable is which turbo LoRA, and it is not a small one.

**Measured from the safetensors headers, not argued:**

| LoRA | modules | touches | rank |
|---|---|---|---|
| official fl2v 8-step | 208 | `qkv_proj`, `out_proj`, `fc1`, `fc2` | 128 / 384 |
| this one (v4 600 ema) | **259** | those **plus 51 `adaln_proj.linear`** | 64, adaln at **16** |

Those 51 extra modules are the 50 per-block `adaln_proj` and
`final_layer.adaln_proj` -- the conditioning-modulation path, which the
official LoRA leaves untouched and this one adapts at a deliberately
separate low rank. (Until 2026-08-20 this note called that path "the place
fl2va and ref2va differ MOST"; withdrawn, the figure behind it compared
curve-form coefficients on differently-signed bases. At the modulation
output the parents differ by a few percent there, as they do everywhere.)

**Why the pack's own two nodes instead of `LoraLoaderModelOnly`.** Our base is
*pruned*. This LoRA's time conditioning has to be re-injected at run time from
a `silu(t_emb)` grid the pack ships. The stock loader applies the weights,
silently skips that, and reports nothing -- a wrong render, not an error.

**`low_vram` is off, and that is deliberate.** On it merges the LoRA into the
weights for a lower peak; its README says merging comes out softer on
quantized bases, and ours is int8 *and* pruned, so we would pay that twice.
It is the dial to reach for on an OOM, not before.

**What this arm is not.** The pack's README claims t2v and i2v and never
mentions ref2va. Running it here is our experiment; a poor result is evidence
about an unsupported combination, not a defect in the LoRA."""


_NOTE_TURBO_PACK_SPLIT = """\
## Base first, distill last -- the variant with an actual prior behind it

Two stages off one `SplitSigmas`: the base checkpoint runs the opening steps,
the turbo LoRA finishes. Its twin is `h3_probe_ref2v_turbo_pack.json`, the
same LoRA with no split.

**The reason to expect this to help is specific.** What diverges between fl2va
and ref2va is concentrated in the conditioning-modulation path -- the
`adaln_proj` family -- and conditioning binds hardest in the EARLY steps,
while composition and identity are still being decided. Late steps are mostly
refinement. So spend undistilled steps where the references are established
and distilled steps where they are only being sharpened.

If that story is right, this arm keeps reference blending that the
single-stage distill loses, at most of the speed. If the single-stage arm
already blends fine, this one costs time for nothing and the story was wrong.
Both outcomes are worth knowing and neither is readable from one arm alone.

**Watch the audio.** Its README calls audio the weaker axis at low step
counts, and these arms carry a `fully_copy` reference track, so a distilled
tail is exactly where lip-sync and continuity would break first."""


def _note_ref_relationship(role: str) -> str:
    what = {
        "swap": ("replacing a character in a source video", """\
**This is the only swap prompt left, and two others were rendered against it
and retired on 2026-08-22.** An imperative arm carried from a community
write-up, and a concise one-paragraph twin written here. The concise arm is
the one that settles the format question this graph used to be half of: the
owner judged it **broken speech, gibberish, 3 of 3** at the shipped canvas and
length on matched seeds, and the log-mel measure ordered it the same way
without hearing anything -- bad in the FIRST third on two of three seeds,
where this arm starts at 0.589-0.704. `bench/results/2026-08-22_swap_prompt_verdict_362.json`
holds the numbers.

**What that does NOT establish is that the six sections are the cause.** This
prompt differs from the retired one in structure AND length AND whether the
soundtrack is stated as an `<Audio 1>: fully_copy` retention line or as prose.
Three variables moved together and the separating arm was never rendered.

**A separate problem is open and is not about the prompt.** This arm drifts in
the last third of a 15.083s render -- 0.704/0.647/0.481 at its best seed,
0.688/0.537/0.017 at its worst -- because a 19.56s source is cut mid-delivery
and 362 frames is the trained ceiling. See the reference-video note below.

This is the **character swap** arm: the video is the *plate* and the image is
the *new identity*. Read it against `h3_ref_video_image_edit`, which is the
same machinery pointed at a different question -- there the person in
`<Video 1>` stays and their garment changes; here the person is the only
thing that changes and everything around them must not.

**Its distinguishing feature is a technique the official guide does not
contain.** `<Picture 1>` and `<Video 1>` are each told what they do *not*
supply:

```
<Picture 1> supplies subject identity only. It does not supply lighting,
    exposure, colour grade, background, camera angle, pose, framing, or
    scene composition.
<Video 1> ... It does not supply the face or identity.
```

The guide never writes a negative clause -- every relationship there is
stated as what a reference *provides*. These come from general prompting
research, where the reported failure is the model blending the two
identities, or dragging the image's own lighting and background into the
plate. **Whether the negatives earn their tokens is untested here**, and it
is the reason this arm exists rather than a claim it ships with.

`[video editing]`, not `[reference generation]`, because the source video is
directly modified -- and `+ audio reuse` alongside it, since the original
track stays audible. Community write-ups of this scenario routinely stop at
a bare `[video editing]`; guide section 3.2 asks for both.

**A reference image that is too small, or a face too far from the camera,
is the failure mode to rule out first.** The identity has to survive being
resized into the reference budget before any of the wording above matters."""),
        "edit": ("editing a source video", """\
This is the **edit / "inpaint over it"** arm, and the first thing to know is
that H3's reference node has **no mask socket**. The edit is whole-frame
regeneration conditioned on the source, not a painted region. What holds the
untouched parts still is `retention_analysis` saying precisely what survives:

```
<Video 1> (source video for the edit): partially_preserved - framing, camera
    movement, and shot timing are kept; only what is named above changes.
<Subject 1> ...: partially_preserved - face, build, posture, and motion are
    retained from <Video 1>; the garment changes.
<Subject 2> ...: attribute_transfer - the red jacket replaces the original
    garment on <Subject 1>.
```

`partially_preserved` is the marker that means "keep this, except". Using
`fully_preserved` here asks for a copy and gives the edit nowhere to happen;
using `weak_reference` throws away the framing you are trying to keep."""),
        "continue": ("continuing from the end of a source video", """\
The **continuation / extend** arm. `<Video 1>` is a starting state rather than
a thing to copy, so the marker is `partially_preserved` on the continuation
relationship and the shot text says plainly that it begins where the source
ends, without a cut.

Worth knowing about the geometry: the reference video is truncated to the
GENERATED frame count and snapped down to the 17n+5 grid, so a continuation is
conditioned on at most as many frames as it will produce. A long source does
not buy a longer run-up."""),
        "motion": ("transferring motion onto a different subject", """\
The **motion transfer** arm, and the one that uses a mechanism the others do
not. Motion does not ride on `<Video N>`: guide section 2.1 defines ONE subject
from TWO assets, naming what each provides.

```
<Subject 1> is the person whose appearance comes from <Picture 1> and whose
    walking motion comes from <Video 1>.
<Subject 1> ...: attribute_transfer - the gait and timing of <Video 1> are
    transferred to the person in <Picture 1>.
```

`attribute_transfer` is defined as "referenced characteristics are transferred
to a different identifiable target subject", which is exactly this. The video's
own scene is explicitly NOT reused, and the definition says so, because
otherwise the model has two competing environments."""),
        "voice": ("referencing a speaker's voice", """\
The **voice timbre** arm. Section 2.4 lists voice as an audio reference use and
requires the target speaker's **global speaker id** in the definition:

```
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).
```

The id comes from the target video's speaker order and is not renumbered for
the audio. The marker is `reference`, from the AUDIO set -- the signal is not
copied, only timbre and delivery. `fully_copy` would ask for the source
waveform itself, which is a different request.

This is the only arm here that puts a spoken line in `overall_soundscape`, so
it is also the only one testing whether the referenced timbre survives into
generated speech."""),
    }[role]
    return f"""\
## Reference relationship: {what[0]}

The five socket-combination arms all ask for the same weak thing -- pacing at
`weak_reference` -- because which sockets are wired is mechanical. **What the
prompt asks those labels to do is the axis that changes the output**, and this
arm isolates one point on it.

{what[1]}

## Markers do not cross sets

Visual labels take `fully_preserved`, `partially_preserved`,
`attribute_transfer`, `weak_reference` (guide 4.1). Audio labels take
`fully_copy`, `partially_copy`, `reference`, `weak_reference` (4.2). Only
`weak_reference` appears in both. `bench/check_ref_prompt_labels.py` checks the
labels exist; it does NOT check you picked a sensible marker, so that part is
on the reader.

See `docs/h3_references.md` for the full reference-type reference.
"""


def _note_ref_matrix(what: str) -> str:
    return f"""\
## Reference matrix arm: {what}

One of five graphs that differ only in **which typed references are appended**.
Run them against each other; everything else -- seed, prompt skeleton, canvas,
length, sampler, attention chain -- is shared by construction.

| graph | images | video | its soundtrack | standalone audio |
|---|---|---|---|---|
| `h3_ref_video_only` | | yes | | |
| `h3_ref_video_audio` | | yes | yes | |
| `h3_ref_image_audio` | yes | | | yes |
| `h3_ref_video_to_video` | yes | yes | yes | |
| `h3_ref_image_video_audio` | yes | yes | yes | yes |

**The prompt in each one declares exactly the labels that graph wires**, and
`bench/check_ref_prompt_labels.py` fails the build if that stops being true.
The numbering is the tokenizer's, not a convention. The typed surface permits
arbitrary list order; this generator deliberately preserves the legacy order:
images, then videos with each owned soundtrack's `<Audio j>` immediately BEFORE
its `<Video k>`, then standalone audio, with a separate 1-based counter per type.
So in the all-types arm the soundtrack is `<Audio 1>` and the standalone clip
is `<Audio 2>`, while the video is `<Video 1>` in every arm that has one.

**A silent clip cannot have a soundtrack pulled.** VHS raises
"failed to extract audio" when its audio output is pulled on a video with no
audio stream, and the render dies at execution having validated cleanly. The
video-only arm therefore loads a different, silent clip and leaves the append
node's soundtrack
alone.

`force_rate` is {REF_VIDEO_FORCE_RATE:g} on every arm that loads a video. See
`h3_ref_video_to_video.json` for why that is not optional.
"""


_NOTE_REF_VIDEO = f"""\
## The first graph here that wires a reference video

Everything this repo knew about reference video before 2026-08-13 was read off
source and never executed. This graph is what executing it looks like.

## force_rate is 24, and it is not optional

The native `ref_videos.ref_video_0` socket takes an **IMAGE batch**, not a
VIDEO. Native ComfyUI has **no fps input at all** and assumes 24 twice over:
once for the DiT's
temporal clock, and once for the `<T.T seconds>` labels the conditioner reads
off the 2 fps subsample. The reference pipeline instead resamples onto 24 from
the rate the container reports, and diffusers' own docstring flags the hazard
in as many words -- a video whose real rate is lost on the way in is
conditioned at the wrong speed, silently.

**Measured**, on three 6.00-second clips trimmed to differ only in frame rate,
with `force_rate=0` against `force_rate={REF_VIDEO_FORCE_RATE:g}`:

| source | frames handed over | snapped to 17n+5 | H3 reads it as | error | last label |
|---|---|---|---|---|---|
| 24 fps | 144 | 141 | 5.875s | 0.0% | `<5.2 seconds>` |
| 25 fps | 150 | 141 | 5.875s | **+4.2%** | `<5.2 seconds>` |
| 30 fps | 180 | 175 | **7.292s** | **+25.0%** | `<7.0 seconds>` |

At 30 fps the model is told a six-second reference is seven and a quarter
seconds of action, and the conditioner's final timestamp says
`<7.0 seconds>` where it should say `<5.2 seconds>`. **A 24 fps source is
unaffected either way**, which is exactly why testing on one proves nothing.

This repo's `MiniMaxH3AppendRefVideo` also owns `VHS_VIDEOINFO` and normalizes
from its `loaded_fps`. Shipped graphs still hold `force_rate=24` so the clock
policy did not change in the ordering migration; `bench/check_ref_prompt_labels.py`
fails the build if a shipped reference-video loader drops it.

## What it costs, and why the video path has no upscale knob

Reference rows ride every sampling step exactly as video rows do. A five-second
reference at the full 1344x768 canvas is **+32,256 rows**, taking the sequence
from 38,222 to 70,478 -- 1.84x, and attention goes as the square, so roughly
3.4x the attention work. A `max` image reference is +7,168 by comparison.

Budget references by pixel area, not by count: the same clip at 640x360 costs
+7,040.

The image path carries `allow_upscale` on its append node because ComfyUI
clamps image references with `min(1.0, 2048/short_edge)` where the reference
pipeline has no clamp. **The video path has the same class of divergence** --
ComfyUI refuses to upscale a reference video, the reference puts it on the full
canvas rule -- and deliberately has no knob closing it. Closing it costs 5x what the
image one does, and nothing has measured whether it buys anything.

## Two more divergences to know about

- **Native reference audio is not truncated.** The reference cuts a soundtrack
  to the generated duration; core encodes the whole waveform, at 80 rows per
  unwanted second. This repo's typed conditioner caps owned and standalone
  audio internally; that handles shipped graphs, not native ComfyUI generally.
- **The frame count snaps DOWN** to the 17n+5 grid after being truncated to the
  generated length, and fewer than 5 frames raises.

## Labels

`<Video k>` and `<Audio j>` are numbered independently, and an owned
soundtrack's `<Audio j>` is emitted immediately BEFORE its `<Video k>`. One
video with sound therefore reads as `<Audio 1>` then `<Video 1>`. Images are
`<Picture i>`. This generated graph appends its images first.

**The shipped clip has an audio track**, owned by its video append record, so
the prompt declares `<Audio 1>`. Swap in a silent clip and remove both the
soundtrack link and those prompt lines. Section 4.2's `<Audio N>` markers are a
different set from the visual ones:

```
subject_definitions:
<Audio 1> is the synchronized audio track of <Video 1> and is reused in the target video.

retention_analysis:
<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.
```

Valid audio markers are `fully_copy`, `partially_copy`, `reference` and
`weak_reference`. Valid visual markers are `fully_preserved`,
`partially_preserved`, `attribute_transfer` and `weak_reference`. They do not
interchange.

Limits diffusers enforces and ComfyUI, sglang, and this typed surface do not:
9 images, 3 videos, 3 audios, **12 references total**, and an audio reference
may never appear without an image or video.
"""


def _note_split(base_last: bool) -> str:
    order = ("distilled student on the high-noise steps, plain base model on "
             "the finish" if base_last else
             "plain base model on the high-noise steps, distilled student on "
             "the finish")
    twin = ("h3_probe_split_base_first.json" if base_last
            else "h3_probe_split_base_last.json")
    why = ("""**Why this ordering.** The distilled student's measured deficit is
high-frequency detail, and high-frequency detail is resolved at low sigma. So
putting the student on the *finishing* steps places its known weakness exactly
where a reference-heavy or identity-heavy render needs the most. Base-last
spends the base model's cost where it buys most and keeps the speedup where
the student is strong."""
           if base_last else
           """**Why this ordering.** This is the Krea 2 arrangement, where the
win was seed and compositional diversity at near-turbo cost: the base model
forms the composition in the high-noise steps and the distilled student
delivers a fast, sharp finish. It is the right way round when the finish is
about sharpness rather than identity.""")
    return f"""\
## Two-stage split: {order}

One `BasicScheduler` feeds `SplitSigmas`, and both halves sample **the same
curve**. That shared schedule is the whole precondition, and it is why both
stages carry the same `ModelSamplingMiniMaxH3` values. Two different shifts
would mean the two halves are integrating different curves and the handoff
means nothing.

Run this against **{twin}**, which is the same graph with the two models
swapped.

{why}

## Sweep the boundary from 1, not from 3

H3's schedule is far more front-loaded than the model this pattern came from.
At video shift 12 and 8 steps the evaluation points are

```
1.0  0.9882  0.973  0.9524  0.9231  0.878  0.8  0.6316
```

Seven of the eight sit at sigma >= 0.8, and the **final interval alone covers
the bottom 63% of the range**. Krea 2's sweet spot of k=2-3 was still at sigma
0.84 there; here k=3 is 0.9524, barely denoised. This graph ships k={SPLIT_AT}.

## Honest caveats

- **Both orderings have a handoff mismatch.** A distilled student's state after
  its steps is not on the base model's trajectory, so whichever model receives
  the handoff gets an input whose sigma label does not match its actual noise
  content. The reverse ordering has the same problem mirrored. Nobody has
  measured this for H3.
- **Two samplers are expressible here and nowhere else.** Each stage has its
  own `KSamplerSelect`, so a multistep base stage into a first-order distilled
  finish is one graph. At low k the base stage has no multistep history yet and
  degenerates to euler, which is exactly where the front-loaded schedule wants
  the boundary -- so that freedom is smallest where it is most wanted.
- `add_noise` is not a widget in this stack. `DisableNoise` is the
  custom-sampler spelling of it, and it is what stage 2 reads.
"""


_NOTE_TURBO_768P = f"""\
## The turbo LoRAs whose shift is not 12/3

This graph loads the **{turbo_label(TURBO_768P_LORA)}** LoRA at
**{TURBO_768P_STEPS} steps, strength {TURBO_768P_STRENGTH:g}**, video shift
**{TURBO_768P_SHIFT["shift_video"]:g}** and audio shift
{TURBO_768P_SHIFT["shift_audio"]:g}.

**Steps and strength here are an owner-selected recipe, not the vendor's
row.** The vendor documents {TURBO_768P_DISTILLED_STEPS} NFE at strength 1.0,
and that is what the student was distilled to do; the table below records it.
Six steps at 0.75 is what the owner's own trials preferred as of 2026-08-23,
and it is provisional -- no blind distribution has scored it. The shift is
NOT part of the recipe: 6/3 is the training value and stays.

| LoRA | trained at | shift (v/a) | steps |
|---|---|---|---|
| 4-step v0.1 | 544p, mixed aspect | 12 / 3 | 4 |
| 8-step v1.0 | 544p, mixed aspect | 12 / 3 | 8 or 4 |
| {turbo_label(TURBO_768P_LORA)} (this graph) | **1344x768** | **6** / 3 | {TURBO_768P_DISTILLED_STEPS} distilled, **{TURBO_768P_STEPS} rendered** |
| ref2v 4-step v0.1 | 544p, mixed aspect | 12 / 3 | 4 |
| 4-step v0.1 768p **SLA** | **1344x768** | **6** / 3 | 4 |

The SLA one shares this graph's row and is its own probe,
`h3_probe_turbo_768p_sla.json`: same graph, only the LoRA file swapped.

**A turbo LoRA inherits the sampler's shift. It does not carry its own.** So
loading this one into a graph whose ModelSamplingMiniMaxH3 still reads 12/3
samples it off a schedule it never saw, and **nothing errors** -- the render
completes and looks plausibly wrong. That is the whole reason this ships as
its own graph instead of a sentence telling you to change two widgets.

`bench/check_distill_settings.py` enforces the pairing across every shipped
graph and grades the table above against the vendor's own README.

**This is the one turbo LoRA that is already home on this canvas.** It was
distilled at 1344x768, which is what this graph renders. The 8-step was
distilled at 544p, so `h3_text_to_video_turbo.json` is the graph with a
resolution tension and this one is not.

The trade is steps: 4 here against the 8-step's 8. Fewer evaluations on a
schedule whose final jump is already the largest one it takes.
"""


def _note_ref_transfer(checkpoint: str, what: str) -> str:
    return f"""\
## Does an fl2v distill transfer to reference work, and on which weights

This is `h3_probe_capture_ref3.json`'s request -- three reference images
(character, garment, environment), one continuous shot, 1024x768 x 362 -- run
on **{checkpoint}** with the 4-step 768p turbo LoRA at the vendor's row
(euler, `simple`, 4 steps, shift 6/3, strength 1.0). {what}

Four graphs share everything but the checkpoint: `fl2va`, the HF hybrid
`b30-49` (fl2va with ref2va's adaln in blocks 30-49), a locally built hybrid
with ref2va's adaln in **all** blocks and the final layer
(`bench/build_hybrid.py`, which first reproduces the HF file byte-for-byte),
and `ref2va` itself.

**The prediction, written before the render.** The lightx2v fl2v LoRAs were
fitted against fl2va's attention and MLP weights. Both hybrids keep those
weights; ref2va's differ from them by about 3% relative
(`bench/results/2026-08-20_dit_internals.json`). If the LoRA's reference
handling holds on the hybrids and breaks on ref2va, that difference is the
mechanism. The all-adaln hybrid adds a second question: if it transfers as
well as ref2va does, the "adaln is the reference pathway" reading holds
functionally; if it does not, the linears matter.

**How to read it: briefs met, never clips matched.** Does the reference
identity survive at all, per checkpoint. Single seed first; a distribution if
the single seed separates the arms. Bench arms patch the LoRA file
(`--set LABEL:LoraLoaderModelOnly.lora_name=...`) to run the v1.1 and SLA
releases on the same four graphs.

**Two confounds carried on purpose.** The canvas is the capture graph's
1024x768, the configuration known to fit three references at full length and
the one the activation captures were taken at; the LoRA's one trained shape is
1344x768. And fl2va-family weights never saw reference rows, so a failure on
the `fl2va` arm is informative rather than a bug.
"""


_NOTE_TURBO_OWNER = f"""\
## The owner's recipe, not the vendor's row

This is `h3_text_to_video_turbo_4step_768p.json` with **three** things moved,
all at once, to the settings the owner arrived at in their own t2v trials on
2026-08-20: sampler `er_sde` -> `{TURBO_SAMPLER}`, scheduler `simple` ->
`{TURBO_OWNER_SCHEDULER}`, LoRA strength 1.0 -> {TURBO_OWNER_STRENGTH:g}.
Because three knobs move, a difference against the vendor graph is not
attributable to any one of them; this graph is a recipe, and it is judged as a
recipe against the vendor-recipe arm in the same blind session.

Two costs the recipe carries, stated up front:

- **Strength below 1.0 at 4 steps under-distills.** The student was fitted at
  1.0 on a 4-step schedule; 0.75 interpolates toward a base model that needs
  16 steps. `docs/h3_ref2v_distillation.md` recommends strength sweeps on the
  8-step LoRA at 6-8 steps for exactly this reason.
- **`beta` halves Sol-Attn's sparse steps here.** Sol's window is a sigma band
  (0.96 down to 0.40 at shift 6 with the shipped 0.2/0.9). Under `simple` 3 of
  4 steps land inside it; under `beta` only 2 of 4, because beta's second
  sigma is 0.966, just above the ceiling. At 6 steps it is 4/6 against 3/6.
  So this recipe runs more of the trajectory dense than the vendor row does,
  which is a speed cost and, for a Sol-quality question, a confound.

Bench arms patch the LoRA file onto this graph (`run_graph_arms.py --set
LABEL:LoraLoaderModelOnly.lora_name=...`) rather than shipping a row per
file, so one graph carries the recipe and the file is the only thing that
moves. **This paragraph argued the opposite until 2026-08-23**: it said only
the v1.0 768p file had a vendor-attested row and that v1.1 was deliberately
not a graph. v1.1 is now the graph -- v1.0 left this machine -- and its 6/3
shift and 4 steps are inherited from v1.0's vendor row by filename family
rather than attested. `bench/check_distill_settings.py::UNATTESTED` is where
that is written down, and it fails if the vendor ever publishes the real
row and the declaration is left standing.
"""


_NOTE_FL2V = f"""\
## First and last frame, and the first in-distribution turbo arm

Two keyframes into one continuous shot. `MiniMaxH3Conditioning` derives the
canvas from the FIRST frame under `from_keyframe`, the way the release does
(`resolve_canvas_size` on `keyframes[0]`), and cover-crops the closing frame to
match. The `width`/`height` in this graph are the FALLBACK, and they govern
only if you switch `canvas` to `explicit`. **Load a 3:2 still to render at
{FL2V_CANVAS['width']}x{FL2V_CANVAS['height']}**; load something else and you
get that image's aspect on H3's grid, which is the point of the mode.

The prompt carries the FL2VA alignment sentence, which is not the I2VA one with
a word changed: it is the only one of the three that carries no angle brackets
and no square brackets (`base_en.md:14-32`). Its `S.SS` is derived from
`length` by `fl2v_prompt()` rather than typed, so it cannot drift from the
graph.

**One shot, deliberately.** `base_en.md:60`: FL2VA "generally favors a single
shot so the model can interpolate continuously from the first frame to the last",
and multiple shots are for when they are explicitly specified.
"""

_NOTE_FL2V_TURBO = f"""\
## The first turbo arm this repo has run in distribution

Every turbo LoRA the vendor released is an **fl2v** distill -- the filenames say
so and so does `_NOTE_REF2V_TURBO` -- and until this graph every turbo arm here
was t2v or ref2v. So every turbo number and every turbo comparison recorded in
this repo was taken out of distribution, including the ones that reason
carefully about *how far* out.

This is `h3_first_last_frame_to_video.json` with the
{turbo_label(TURBO_768P_LORA)} LoRA at this repo's current recipe:
{TURBO_768P_STEPS} steps, shift 6/3, `{TURBO_SAMPLER}`, strength
{TURBO_768P_STRENGTH:g}. The shift is the vendor's training value; the steps
and strength are owner-selected and provisional, so a number taken here is
comparable to other arms on this recipe and not to the vendor's
{TURBO_768P_DISTILLED_STEPS}-NFE row. Same canvas, same seed, same length, same placeholder
keyframes as its base twin, so the pair is comparable by construction.

**What it is for.** Not "is turbo good" -- a rendered pair cannot A/B a
numerical knob, and this repo's own rule says so. It is the missing reference
point: when a ref2v turbo arm degrades, there has been no in-distribution arm to
say how much of that is the LoRA being asked to do something it was not
distilled for. Now there is one.

**The short edge matters here.** {FL2V_CANVAS['width']}x{FL2V_CANVAS['height']}
carries the 768 short edge this LoRA is named for, unlike the 544p home canvas.
"""


_NOTE_REF2V_TURBO = f"""\
## Deliberately out of distribution

This is `h3_image_ref_plus_text_to_video.json` with an **fl2v** turbo LoRA
loaded onto the **ref2va** checkpoint. That pairing is not supported and is
not meant to be: all three released turbo LoRAs are `fl2v`, and the vendor
lists ref2v distillation as unshipped future work.

It is here because how it fails is informative, and because the failure is
silent. The two checkpoints have **identical tensor key sets**, so the LoRA
applies with zero unmatched keys and no warning.

**What the numbers say to expect** (see `docs/h3_ref2v_distillation.md`):

- ref2v is a separate `transformer_ref` partition measuring **4.2%** relative
  Frobenius from fl2va. The whole 8-step turbo LoRA measures **0.036%**. The
  distillation target moved about 120x further than the adapter reaches.
- The LoRA touches only `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1` and
  `mlp.fc2`. It does **not** touch `final_layer`, `adaln_proj`, the norms or
  the patch projections. (A claim that those are "where the two checkpoints
  differ most" was withdrawn 2026-08-20; they differ by a few percent there,
  as the linears do.) So expect degradation, not garbage. NaN or noise means
  a wiring error, not this.
- fl2v conditioning sits at the target's own rotary coordinates; a reference
  does not, and pushes the target's origin by 1 to 1206 units.

**Look for identity drift, not collapse.** The subject stays the right kind of
thing in roughly the right clothes; what goes is the specific face, the
hairline, logo text, fabric weave. Compare a still against the reference at
100%.

**The diagnostic test:** re-run with the reference order reversed. If the same
reference behaves differently at slot 1 than at the end, that is the rotary
coupling rather than generic quality loss -- fl2v cannot produce that
signature.

**Knobs, in order of expected payoff.** Lower the LoRA strength first: {TURBO_LORA_STRENGTH:g}
is shipped here, and public in-distribution evaluation needed 0.75 even on the
model the LoRA was trained for. Use **0.01, not 0.0**, as the control -- 0.0
short-circuits the dequantise/add/requantise round trip entirely and is not a
like-for-like baseline. Then try a two-stage split, and note the ordering:
**base-last**, not base-first. The distilled student's measured weakness is
high-frequency detail, resolved at low sigma, and high-frequency identity is
the entire point of a reference. **Leave the shift at 12/3.**
"""


_NOTE_TURBO = f"""\
## Turbo LoRA: what the training resolution means

This graph loads the **8-step v1.0** LoRA at {TURBO_STEPS} steps, shift
{TURBO_SHIFT["shift_video"]:g}/{TURBO_SHIFT["shift_audio"]:g}.

| LoRA | trained at | shift (v/a) | steps |
|---|---|---|---|
| 4-step v0.1 | 544p, **mixed aspect** | 12 / 3 | 4 |
| 8-step v1.0 (this graph) | 544p, **mixed aspect** | 12 / 3 | 8 or 4 |
| {turbo_label(TURBO_768P_LORA)} | **1344x768** | **6** / 3 | {TURBO_768P_DISTILLED_STEPS} distilled, {TURBO_768P_STEPS} rendered |
| ref2v 4-step v0.1 | 544p, **mixed aspect** | 12 / 3 | 4 |
| 4-step v0.1 768p SLA | **1344x768** | **6** / 3 | 4 |

**Two things move with the LoRA, and only one of them is the shift.** Steps
always move: 16 is a base-model number. The shift moves only for the 768p
ones, which were distilled at video shift 6. The 544p ones were distilled at
12/3, which is already the default, so for them the shift node stays put.
Changing one without the other is not a partial fix.

## The resolution question

A step-distillation LoRA learns to take bigger jumps along the schedule *at
the token count it saw*. 544p and 768p are roughly a factor of two apart in
tokens, so a 544p LoRA rendering at 1344x768 is working at about twice the
sequence length it was distilled on.

**You cannot satisfy both distributions at once, and that is the real
choice here.** MiniMax H3's own canvas rule is a 768 short edge with a
1344x768 area cap: that is what `adapt_canvas` enforces and what the
reference generates. 544p is below it. So:

- Render at **1344x768** and the base model is in its trained canvas while
  the 544p LoRA is off its distillation resolution.
- Render at **544p** and the LoRA is home while the base model is outside
  the canvas family it was trained on. Nothing stops you: the width and
  height on the conditioning node are plain ints at 32-px steps, so 544p is
  typeable. `MiniMaxH3KeyframeCanvas` is the node that refuses, which is why
  this graph is t2v.

**Which one costs less is not measured here.** Do not assume the LoRA's
resolution wins just because the LoRA is the thing you added.

**The {turbo_label(TURBO_768P_LORA)} is the only one with no resolution gap at this
canvas** -- it was distilled at exactly 1344x768. The trade is aspect: it saw
that one shape, where **both** 544p LoRAs (v0.1 and the 8-step v1.0 this
graph loads) saw mixed aspect ratios. So render 1:1 or 9:16 and the 768p LoRA
becomes the off-distribution one while this graph's LoRA is at home on shape
and away on resolution. Neither is free; they are away in different
directions. `h3_text_to_video_turbo_4step_768p.json` is the sibling to
compare against.

Specs from `coderef/Minimax-H3-Turbo`, README model table.
"""




def _plain_chain_ui(g, unet_node, *, sh, sage, sol, head_chunks,
                    sol_enabled=True):
    """The UI twin of `_plain_model_chain`: a second model path, no LoRA.

    Same UNETLoader, same shift, same attention chain. The shift MUST match
    stage 1's: both halves read sigmas from one `BasicScheduler`, so two
    different shifts would have them integrating different curves and the
    handoff would mean nothing.
    """
    src = g.add("MiniMaxH3SigmaShift", (-1500, 900), size=(360, 110),
                widgets=[sh["shift_video"], sh["shift_audio"]],
                inputs=[_in("model", "MODEL")], outputs=[_out("MODEL", "MODEL")],
                title="Sigma shift (stage 2, must match stage 1)")
    g.link(unet_node, 0, src, "model", "MODEL")
    if sage:
        node = g.add("MiniMaxH3SageAttention", (-880, 900), size=(360, 110),
                     widgets=[SAGE_NODE["mode"], SAGE_NODE["patch_token_refiner"],
                              SAGE_NODE["head_chunks"] if head_chunks is None
                              else head_chunks],
                     inputs=[_in("model", "MODEL")], outputs=[_out("MODEL", "MODEL")])
        g.link(src, 0, node, "model", "MODEL")
        src = node
    if sol is not None:
        node = g.add(SOL_NODE, (-880, 1040), size=(360, 330),
                     widgets=_sol_widgets(sol),
                     # `tau_profile` is no longer a top-level socket: the v3
                     # node moved it inside the "adaptive tau" option, where
                     # the frontend names it `selection.tau_profile`. It is
                     # unwired in every graph here, so it is dropped rather
                     # than renamed to a socket nothing connects.
                     inputs=[_in("model", "MODEL")],
                     outputs=[_out("MODEL", "MODEL")],
                     title=("Patch Sol-Attn (stage 2)" if sol_enabled
                            else "Patch Sol-Attn (stage 2, bypassed)"))
        # Bypass here too, or the split graphs ship Sol enabled on their second
        # model path while every other graph has it off -- and the UI/API
        # cross-check catches it as a node-set mismatch rather than as the
        # policy break it actually is.
        if not sol_enabled:
            g._node(node)["mode"] = 4
        g.link(src, 0, node, "model", "MODEL")
        src = node
    node = g.add("SageChainAssert", (-480, 900), size=(360, 130),
                 widgets=[sage, sage, sage, not sage],
                 inputs=[_in("model", "MODEL")], outputs=[_out("model", "MODEL")],
                 title="Assert the stage-2 chain composed")
    g.link(src, 0, node, "model", "MODEL")
    return node


def build_ui(task: str, *, sage: bool = True, prompt: str | None = None,
             steps: int | None = None, shift: dict | None = None,
             sampler_name: str | None = None, scheduler_name: str | None = None,
             head_chunks: int | None = None, ref_upscale: bool = True,
             ref_video_policy: str = "encoder",
             ref_image_policy: str = "comfy",
             ref_qwen_short_edge: int = 0,
             ref_video: bool = False, ref_video_audio: bool = True,
             ref_images_on: bool = True, ref_image_count: int = 2,
             ref_images: tuple[str, ...] | None = None,
             turbo_pack: bool = False,
             pdd: bool = False,
             pdd_heads: bool = True,
             pdd_nfe: int = 0,          # an override; see the API builder
             ref_audio: bool = False,
             split_at: int | None = None,
             split_base_last: bool = True,
             single_frame: bool = False,
             cache: dict | None = None,
             variant_note: str | None = None,
             length: int = LENGTH, seed: int = SEED, preview: bool = False,
             sol: dict | None = None, sol_enabled: bool = True,
             canvas_mode: str = "match_keyframe", stamp: bool = False,
             last_frame: bool = False,
             unet: str | None = None, lora: tuple[str, float] | None = None,
             out_prefix: str | None = None, title: str | None = None,
             sla_router: float | None = None,
             vae_encoder: str | None = None,
             clip: str | None = None,
             **canvas) -> dict:
    ref = task == "r2v"
    # The same consistency guard `build_api` carries, and it has to be here
    # too: `main()` writes every UI graph in one loop BEFORE the API loop runs,
    # so a guard only in `build_api` lets a wrong `.json` reach disk and then
    # exits -- leaving a graph that loads the one-frame VAE for a 124-frame
    # clip, which is exactly what the guard exists to prevent.
    _check_single_frame(single_frame, length)
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
    cv = dict(CANVAS, **canvas)
    # THE DEFAULT PROMPT FOLLOWS THE SOCKETS, NOT THE TASK STRING. `i2v`
    # covers both keyframe modes -- one wired frame or two -- and they take
    # DIFFERENT alignment sentences (`base_en.md:14-32`), so keying this on
    # `task` alone hands an fl2va graph the I2VA line. Nothing downstream
    # would catch it: preflight checks that the preamble names a Picture,
    # not that it is the right sentence for the mode.
    if prompt is None:
        prompt = (fl2v_prompt(length) if (task == "i2v" and last_frame)
                  else {"t2v": T2V_PROMPT, "i2v": I2V_PROMPT,
                        "r2v": R2V_PROMPT}[task])
    g = UIGraph()

    unet_node = g.add("UNETLoader", (-1500, 0), size=(560, 90),
                      widgets=[unet or MODELS["unet_ref2va" if ref else "unet_fl2va"],
                               "default"],
                      outputs=[_out("MODEL", "MODEL")])
    if clip in CORE_LOADED_ENCODERS:
        clip = g.add(
            "CLIPLoader", (-1500, 140), size=(560, 110),
            widgets=[clip, "minimax", "default"],
            outputs=[_out("CLIP", "CLIP")],
            title="Load H3 encoder (core CLIPLoader, ComfyUI-native file)",
        )
    else:
        clip = g.add(
            "MiniMaxH3AWQEncoderLoader", (-1500, 140), size=(560, 110),
            widgets=[clip or MODELS["clip"], "default"],
            outputs=[_out("CLIP", "CLIP")],
            title="Load custom H3 W4A16 encoder (repo adapter)",
        )
    # Single-frame swaps the decoder, and the node TITLE carries the warning:
    # it is the only thing visible when someone copies this node into a video
    # graph, which is the mistake worth making hard to make.
    vvae = g.add("VAELoader", (-1500, 300), size=(560, 70),
                 widgets=[IMAGE_VAE if single_frame else MODELS["video_vae"]],
                 outputs=[_out("VAE", "VAE")],
                 title=("Load VAE (SINGLE IMAGE ONLY -- do not use for video)"
                        if single_frame else "Load VAE (video)"))
    # Optional encoder promotion; see the note in `build_api`. Only the
    # conditioning node moves onto it -- `VAEDecode` keeps the raw loader, so
    # the graph reads as "encoder changed, decoder untouched".
    vae_enc_src = vvae
    if vae_encoder:
        vae_enc_src = g.add(
            "MiniMaxH3VAEPrecision", (-900, 300), size=(340, 100),
            widgets=[vae_encoder, "unchanged"],
            inputs=[_in("vae", "VAE")], outputs=[_out("VAE", "VAE")],
            title="VAE precision (encode/decode split)")
        g.link(vvae, 0, vae_enc_src, "vae", "VAE")

    avae = g.add("VAELoader", (-1500, 410), size=(560, 70),
                 widgets=[MODELS["audio_vae"]], outputs=[_out("VAE", "VAE")],
                 title="Load VAE (audio)")

    model_src = unet_node
    if lora is not None:
        # Before the attention patches -- see the matching note in build_api.
        # The strength widget is the one thing this graph exists to be swept,
        # so the node gets a title that says what its arm is.
        # See build_api: the turbo pack's loader is not interchangeable with
        # the stock one on a pruned base. Its widget list is three long
        # (lora_name, strength, low_vram) -- the pack's own shipped example
        # graph carries only two, because low_vram was added after it was
        # written, so that example is not the thing to copy the shape from.
        if pdd:
            lora_node = g.add(
                "MiniMaxH3PDDLoRA", (-1500, 560), size=(560, 140),
                widgets=[lora[0], lora[1], pdd_heads, pdd_nfe],
                inputs=[_in("model", "MODEL")],
                outputs=[_out("MODEL", "MODEL")],
                title=(f"PDD LoRA (strength {lora[1]}"
                       + (f", {pdd_nfe} NFE" if pdd_nfe else "")
                       + ("" if pdd_heads else ", HEADS OFF -- control arm")
                       + ")"))
        else:
            lora_node = (
                g.add("MiniMaxH3TurboLoRA", (-1500, 560), size=(560, 140),
                      widgets=[lora[0], lora[1], TURBO_PACK_LOW_VRAM],
                      inputs=[_in("model", "MODEL")],
                      outputs=[_out("MODEL", "MODEL")],
                      title=f"Turbo LoRA (pack node, strength {lora[1]})")
                if turbo_pack else
                g.add("LoraLoaderModelOnly", (-1500, 560), size=(560, 110),
                      widgets=[lora[0], lora[1]],
                      inputs=[_in("model", "MODEL")],
                      outputs=[_out("MODEL", "MODEL")],
                      title=f"Load LoRA (ref delta, strength {lora[1]})"))
        g.link(unet_node, 0, lora_node, "model", "MODEL")
        model_src = lora_node

    # See the matching note in build_api. Titled with its values because the
    # whole reason it is in the graph is that a turbo LoRA needs them changed,
    # and a node showing "ModelSamplingMiniMaxH3" and nothing else does not
    # prompt anyone to look.
    sh = shift if shift is not None else SIGMA_SHIFT
    sigma_node = g.add("MiniMaxH3SigmaShift", (-1500, 700), size=(360, 110),
                       widgets=[sh["shift_video"], sh["shift_audio"]],
                       inputs=[_in("model", "MODEL")],
                       outputs=[_out("MODEL", "MODEL")],
                       title=f"Sigma shift (video {sh['shift_video']:g}, "
                             f"audio {sh['shift_audio']:g})")
    g.link(model_src, 0, sigma_node, "model", "MODEL")
    model_src = sigma_node

    sage_node = None
    if sla_router is not None:
        if sage or sol is not None:
            raise SystemExit("sla_router is an alternative to the sage+Sol chain; "
                             "pass sage=False and sol=None with it")
        router_node = g.add("MiniMaxH3SLARouter", (-880, -140), size=(360, 90),
                            widgets=[sla_router],
                            inputs=[_in("model", "MODEL")], outputs=[_out("MODEL", "MODEL")],
                            title="SLA top-k router (comparative arm)")
        g.link(model_src, 0, router_node, "model", "MODEL")
        model_src = router_node
    if sage:
        sage_node = g.add("MiniMaxH3SageAttention", (-880, 0), size=(360, 110),
                          widgets=[SAGE_NODE["mode"],
                                   SAGE_NODE["patch_token_refiner"],
                                   SAGE_NODE["head_chunks"] if head_chunks is None
                                   else head_chunks],
                          inputs=[_in("model", "MODEL")],
                          outputs=[_out("MODEL", "MODEL")])
        g.link(model_src, 0, sage_node, "model", "MODEL")
        model_src = sage_node

    if sol is not None:
        # After sage, never before. SolAttn composes by walking the model's
        # existing object patches and wrapping the attention forwards it
        # finds; run first it has nothing to find, and ours then overwrites
        # its patch. Both orders load and render, which is exactly why it is
        # worth pinning in a generated graph instead of leaving to hand-wiring.
        #
        # Enabled when the graph is built for it, bypassed otherwise. Bypass
        # passes MODEL straight through, so a graph carrying a disabled
        # Sol-Attn node still loads and renders without the node installed.
        # The error-prone part is the ordering above, not the toggle.
        sol_node = g.add(SOL_NODE, (-880, 190), size=(360, 330),
                         widgets=_sol_widgets(sol),
                         # tau_profile, added by Sol-Attn 0e334dc: per-block tau
                         # overriding the base value. It is declared
                         # `force_input=True`, so it is a SOCKET, not a widget.
                         # An earlier version of this file emitted it as a 13th
                         # widget value instead. That was harmless in effect --
                         # it landed after dense_blocks, and LiteGraph drops
                         # widget values past the end of the widget list -- but it
                         # meant the node carried a widget count no build of
                         # Sol-Attn has ever had, and the socket was never
                         # declared at all.
                         #
                         # The API-graph validator cannot catch this class of bug:
                         # API graphs have no widget list, so widget/socket
                         # confusion is invisible there. That is what
                         # check_workflow_schema.py is for.
                         #
                         # The v3 node (2026-08-22) moved it inside the
                         # "adaptive tau" option, so its socket is now named
                         # `selection.tau_profile`. Dropped rather than
                         # renamed: one tau everywhere is what we ship, and it
                         # was unconnected under the old name too.
                         inputs=[_in("model", "MODEL")],
                         outputs=[_out("MODEL", "MODEL")],
                         # Titled with the derived value when there is one, for
                         # the same reason MiniMaxH3SigmaShift is titled with
                         # its shifts: `end_percent` is populated from the
                         # graph's STEP COUNT, and a node showing "Patch
                         # Sol-Attn" and a bare 0.87 does not prompt anyone to
                         # ask why it is not 0.9 -- or warn them that editing
                         # `steps` by hand leaves it stale, which nothing at
                         # run time will say. See
                         # h3_config.SOL_END_PERCENT_BY_STEPS.
                         title=_sol_title(sol, sol_enabled))
        if not sol_enabled:
            g._node(sol_node)["mode"] = 4
        g.link(model_src, 0, sol_node, "model", "MODEL")
        model_src = sol_node

    prev_node = None
    if preview:
        # The largest practical saving on a long clip, and not a kernel
        # change: a 362-frame render is ~17 min, so seeing step 3 is what
        # lets a bad seed die at 90 s instead of costing the whole run.
        #
        # It has to be this node rather than ComfyUI's built-in preview,
        # because the launcher passes --preview-method none globally; this
        # node sidesteps that by pushing its own frame to a DOM widget on
        # itself. taeh3 is the H3 tiny decoder (latent_channels 24,
        # patch_size 2) -- without it H3 has no approx VAE at all and
        # previews degrade to latent2rgb.
        #
        # preview_frames=4 rather than 1: a still frame catches a bad
        # composition, but the failures worth aborting a 17-minute render
        # for are motion failures, and those need more than one frame.
        prev_node = g.add("ModelPreviewOverrideKJ", (-460, 190), size=(360, 200),
                          widgets=[512, 80, True, 4, 8, "taeh3.safetensors"],
                          inputs=[_in("model", "MODEL"),
                                  _in("vae", "VAE", optional=True)],
                          outputs=[_out("MODEL", "MODEL")],
                          title="Preview (taeh3)")
        g.link(model_src, 0, prev_node, "model", "MODEL")
        model_src = prev_node

    # See build_api: geometry comes from Resolution everywhere except i2v,
    # where the keyframe decides it.
    resn = None
    if task != "i2v":
        rw = _resolution_widgets(cv["width"], cv["height"], length)
        order = ["shape"] + [k for k in rw if k not in ("shape", "length")] + ["length"]
        resn = g.add("MiniMaxH3Resolution", (-1900, 900), size=(400, 200),
                     widgets=[rw[k] for k in order],
                     outputs=[_out("width", "INT"), _out("height", "INT"),
                              _out("length", "INT"), _out("video_tokens", "INT"),
                              _out("tokens_per_frame", "INT"),
                              _out("attn_cost_vs_16_9", "FLOAT"),
                              _out("summary", "STRING")],
                     title="Resolution: shape, and what it costs")

    img_a = img_b = None
    if ref:
        slots = _ref_image_slots(ref_images_on, ref_image_count, ref_images)
        cond_inputs = [
            _in("clip", "CLIP"), _in("vae", "VAE"), _in("audio_vae", "VAE"),
            _in("references", "MINIMAX_H3_REFERENCES"),
        ]
        cond = g.add("MiniMaxH3ReferenceConditioning", (-460, 0), size=(430, 620),
                     # The True is the ignored legacy vendor_tokens slot. Keep
                     # it before video_policy because saved UI graphs map
                     # widget values positionally, and keep image_policy last
                     # for the same reason -- it is the newest append.
                     widgets=[prompt, cv["width"], cv["height"], length, True,
                              ref_video_policy, ref_image_policy],
                     inputs=cond_inputs + [
                         _in("width", "INT", widget=True), _in("height", "INT", widget=True),
                         _in("length", "INT", widget=True)],
                     outputs=[_out("positive", "CONDITIONING"), _out("LATENT", "LATENT")])
        g.link(vae_enc_src, 0, cond, "vae", "VAE")
        g.link(avae, 0, cond, "audio_vae", "VAE")
        # One typed append node per image, carrying its own sizing. `max`
        # sizes from `short_edge` rather than the target canvas area, which is
        # the policy the shipped graphs have always used.
        #
        # Loaders, then the append chain. Keeping each layer aligned makes
        # presentation order legible in the saved UI graph.
        def row_y(i):
            return 900 + 370 * i

        loads = [g.add("LoadImage", (-1420, row_y(i)), size=(290, 330),
                       widgets=[fname, "image"],
                       outputs=[_out("IMAGE", "IMAGE"), _out("MASK", "MASK")])
                 for i, (_ld, _ft, fname) in enumerate(slots)]
        if loads:
            img_a = loads[0]
        if len(loads) > 1:
            img_b = loads[1]
        # No fit node here either. This is the UI half of the same fold, and
        # `docs/comfy_notes.md` is why both halves move in one commit: nothing
        # checks that the two emission paths agree, so a change to one is
        # invisible until somebody opens a graph.
        chain = None
        for i, src in enumerate(loads):
            append_inputs = [_in("image", "IMAGE")]
            if chain is not None:
                append_inputs.append(
                    _in("references", "MINIMAX_H3_REFERENCES", optional=True))
            append = g.add("MiniMaxH3AppendRefImage", (-760, row_y(i)),
                           size=(280, 150),
                           # positional: size_policy, allow_upscale,
                           # short_edge, qwen_short_edge. `references` is a
                           # socket and consumes no widget slot.
                           #
                           # ALWAYS emit qwen_short_edge, even at 0. build_api
                           # omits it at 0 on the reasoning that 0 is the
                           # node's default and writing it would touch every
                           # reference graph for nothing -- which is correct
                           # THERE, because the API form is keyed by name and
                           # an absent key takes the schema default. It is
                           # invalid here: the UI form matches widget values
                           # BY POSITION, so an omitted value does not default,
                           # it shifts every later widget up one slot. That
                           # reasoning was copied across the format boundary on
                           # 2026-08-25 and corrupted 40 UI graphs until
                           # 2026-08-26.
                           widgets=["max", ref_upscale, _ref_short_edge(),
                                    ref_qwen_short_edge],
                           inputs=append_inputs,
                           outputs=[_out("references", "MINIMAX_H3_REFERENCES")],
                           title=f"Append Picture {i + 1}")
            g.link(src, 0, append, "image", "IMAGE")
            if chain is not None:
                g.link(chain, 0, append, "references", "MINIMAX_H3_REFERENCES")
            chain = append
        if ref_video:
            # See the matching note in build_api. force_rate=24 is the whole
            # point during migration: changing source clock policy at the same
            # time would confound output comparisons. The typed compiler also
            # receives video_info and derives the effective source fps.
            video_y = row_y(len(slots))
            vid = g.add("VHS_LoadVideo", (-1420, video_y), size=(340, 500),
                        widgets={"video": (PLACEHOLDER_VIDEO if ref_video_audio
                                           else PLACEHOLDER_VIDEO_SILENT),
                                 "force_rate": REF_VIDEO_FORCE_RATE,
                                 "custom_width": 0, "custom_height": 0,
                                 "frame_load_cap": length,
                                 "skip_first_frames": 0,
                                 "select_every_nth": 1, "format": "AnimateDiff"},
                        outputs=[_out("IMAGE", "IMAGE"), _out("frame_count", "INT"),
                                 _out("audio", "AUDIO"), _out("video_info", "VHS_VIDEOINFO")],
                        title="Reference video (force_rate 24)")
            append_inputs = [_in("frames", "IMAGE"),
                             _in("video_info", "VHS_VIDEOINFO")]
            if ref_video_audio:
                append_inputs.append(_in("soundtrack", "AUDIO", optional=True))
            if chain is not None:
                append_inputs.append(
                    _in("references", "MINIMAX_H3_REFERENCES", optional=True))
            append = g.add("MiniMaxH3AppendRefVideo", (-760, video_y),
                           size=(280, 150), inputs=append_inputs,
                           outputs=[_out("references", "MINIMAX_H3_REFERENCES")],
                           title="Append video + owned soundtrack")
            g.link(vid, 0, append, "frames", "IMAGE")
            g.link(vid, 3, append, "video_info", "VHS_VIDEOINFO")
            if ref_video_audio:
                g.link(vid, 2, append, "soundtrack", "AUDIO")
            if chain is not None:
                g.link(chain, 0, append, "references", "MINIMAX_H3_REFERENCES")
            chain = append
        if ref_audio:
            audio_y = row_y(len(slots) + int(ref_video))
            aud = g.add("LoadAudio", (-1420, audio_y), size=(300, 130),
                        widgets=[PLACEHOLDER_AUDIO],
                        outputs=[_out("AUDIO", "AUDIO")],
                        title="Standalone audio reference")
            append_inputs = [_in("audio", "AUDIO")]
            if chain is not None:
                append_inputs.append(
                    _in("references", "MINIMAX_H3_REFERENCES", optional=True))
            append = g.add("MiniMaxH3AppendRefAudio", (-760, audio_y),
                           size=(280, 100), inputs=append_inputs,
                           outputs=[_out("references", "MINIMAX_H3_REFERENCES")],
                           title="Append standalone audio")
            g.link(aud, 0, append, "audio", "AUDIO")
            if chain is not None:
                g.link(chain, 0, append, "references", "MINIMAX_H3_REFERENCES")
            chain = append
        if chain is None:
            raise SystemExit("r2v graph has no references to condition on")
        g.link(chain, 0, cond, "references", "MINIMAX_H3_REFERENCES")
    else:
        # Widget order mirrors the schema: prompt, width, height, length,
        # canvas, then the ignored legacy vendor_tokens slot. Keep its True so
        # saved widget positions stay stable. The keyframe images stay sockets.
        cond_inputs = [_in("clip", "CLIP"), _in("vae", "VAE"),
                       _in("first_frame", "IMAGE", optional=True),
                       _in("last_frame", "IMAGE", optional=True),
                       _in("width", "INT", widget=True),
                       _in("height", "INT", widget=True),
                       _in("length", "INT", widget=True)]
        cond = g.add("MiniMaxH3Conditioning", (-460, 0), size=(430, 620),
                     widgets=[prompt, cv["width"], cv["height"], length,
                              ("from_keyframe"
                               if task == "i2v" and canvas_mode == "match_keyframe"
                               else "explicit"),
                              True],
                     inputs=cond_inputs,
                     outputs=[_out("positive", "CONDITIONING"), _out("LATENT", "LATENT")])
        g.link(vae_enc_src, 0, cond, "vae", "VAE")
        if task == "i2v":
            # Straight into the conditioning node. There is no canvas node in
            # front of it any more: it derives the canvas from this keyframe
            # itself, which is one owner instead of two agreeing by wiring.
            img_a = g.add("LoadImage", (-880, 900), size=(290, 330),
                          widgets=[PLACEHOLDER_IMAGE_A, "image"],
                          outputs=[_out("IMAGE", "IMAGE"), _out("MASK", "MASK")])
            g.link(img_a, 0, cond, "first_frame", "IMAGE")
            if last_frame:
                # Mirrors `build_api`. `cross_check` is what asserts the
                # two builders agree, so a second frame added to one and
                # not the other fails at build time rather than at
                # render time.
                img_b = g.add("LoadImage", (-880, 1260), size=(290, 330),
                              widgets=[PLACEHOLDER_IMAGE_B, "image"],
                              outputs=[_out("IMAGE", "IMAGE"),
                                       _out("MASK", "MASK")])
                g.link(img_b, 0, cond, "last_frame", "IMAGE")
    g.link(clip, 0, cond, "clip", "CLIP")

    noise = g.add("RandomNoise", (40, 0), size=(300, 110), widgets=[seed, "randomize"],
                  outputs=[_out("NOISE", "NOISE")])
    samp = (g.add("MiniMaxH3TurboSampler", (40, 150), size=(300, 60),
                  outputs=[_out("SAMPLER", "SAMPLER")],
                  title="Turbo Sampler (pack node)")
            if turbo_pack else
            g.add("KSamplerSelect", (40, 150), size=(300, 60),
                  widgets=[sampler_name or SAMPLING["sampler"]],
                  outputs=[_out("SAMPLER", "SAMPLER")]))
    sched = g.add("BasicScheduler", (40, 250), size=(300, 130),
                  widgets=[scheduler_name or SAMPLING["scheduler"],
                           steps if steps is not None else SAMPLING["steps"],
                           SAMPLING["denoise"]],
                  inputs=[_in("model", "MODEL")], outputs=[_out("SIGMAS", "SIGMAS")])
    guider = g.add("BasicGuider", (40, 420), size=(300, 70),
                   inputs=[_in("model", "MODEL"), _in("conditioning", "CONDITIONING")],
                   outputs=[_out("GUIDER", "GUIDER")])
    sampler = g.add("SamplerCustomAdvanced", (400, 0), size=(320, 150),
                    inputs=[_in("noise", "NOISE"), _in("guider", "GUIDER"),
                            _in("sampler", "SAMPLER"), _in("sigmas", "SIGMAS"),
                            _in("latent_image", "LATENT")],
                    outputs=[_out("output", "LATENT"), _out("denoised_output", "LATENT")])
    vdec = g.add("VAEDecode", (780, 0), size=(260, 60),
                 inputs=[_in("samples", "LATENT"), _in("vae", "VAE")],
                 outputs=[_out("IMAGE", "IMAGE")])
    # No audio decoder on the single-frame path: one frame's share of the
    # audio stream is 0.04s of nothing. Omitted rather than bypassed, so the
    # graph does not carry a node whose presence implies a soundtrack.
    adec = (None if single_frame else
            g.add("VAEDecodeAudio", (780, 110), size=(260, 60),
                  inputs=[_in("samples", "LATENT"), _in("vae", "VAE")],
                  outputs=[_out("AUDIO", "AUDIO")]))
    # One node for mux + save. Its widgets_values is a *dict*, not the
    # positional list every other node uses -- VHS adds format-dependent
    # widgets (pix_fmt, crf, ...) after `format`, so position cannot address
    # them. Shape copied from a frontend-written graph rather than guessed.
    save = (g.add("SaveImage", (1080, 0), size=(500, 560),
                  widgets=[out_prefix or "Image/h3_image_edit"],
                  inputs=[_in("images", "IMAGE")],
                  title="Save the edited image")
            if single_frame else
            g.add("VHS_VideoCombine", (1080, 0), size=(600, 520),
                  widgets={"frame_rate": FPS, "loop_count": 0,
                           "filename_prefix": out_prefix or f"Video/h3_{task}",
                           "format": VIDEO_FORMAT, "pix_fmt": "yuv420p",
                           "crf": 19, "save_metadata": True,
                           "trim_to_audio": False,
                           "pingpong": False, "save_output": True},
                  inputs=[_in("images", "IMAGE"),
                          _in("audio", "AUDIO", optional=True),
                          _in("meta_batch", "VHS_BatchManager", optional=True),
                          _in("vae", "VAE", optional=True)],
                  outputs=[_out("Filenames", "VHS_FILENAMES")]))

    # See the matching note in build_api: last in the chain, asserting the
    # composition rather than any single node's intent.
    assert_node = g.add("SageChainAssert", (-480, 0), size=(360, 130),
                        widgets=[sage, sage, sage, not sage],
                        inputs=[_in("model", "MODEL")],
                        outputs=[_out("model", "MODEL")],
                        title="Assert the attention chain composed")
    g.link(model_src, 0, assert_node, "model", "MODEL")
    model_src = assert_node

    if cache is not None:
        # Mirrors the build_api insertion: after the assert, because the cache
        # is a forward-skipping wrapper over the finished attention chain, not
        # a member of it. Widget order is the node's declared input order.
        cache_node = g.add(CACHE_NODE_CLASS, (-480, 200), size=(360, 150),
                           widgets=[cache["reuse_threshold"],
                                    cache["start_percent"],
                                    cache["end_percent"],
                                    cache["verbose"]],
                           inputs=[_in("model", "MODEL")],
                           outputs=[_out("MODEL", "MODEL")],
                           title="EasyCache: reuse near-identical steps")
        g.link(model_src, 0, cache_node, "model", "MODEL")
        model_src = cache_node

    # The second model path for a two-stage split: same UNETLoader, same
    # shift, no LoRA. Built here rather than lower down because the stage-1
    # guider has to be linked to the right chain the first time -- there is no
    # re-linking in this writer.
    plain_src = None
    if split_at:
        if lora is None:
            raise SystemExit("split_at needs a `lora`; see build_api")
        plain_src = _plain_chain_ui(g, unet_node, sh=sh, sage=sage, sol=sol,
                                    sol_enabled=sol_enabled,
                                    head_chunks=head_chunks)
    stage1_src = model_src
    if split_at and not split_base_last:
        # base_first: the plain base model runs the high-noise steps.
        stage1_src = plain_src
    g.link(stage1_src, 0, sched, "model", "MODEL")
    g.link(stage1_src, 0, guider, "model", "MODEL")
    if resn is not None:
        g.link(resn, 0, cond, "width", "INT")
        g.link(resn, 1, cond, "height", "INT")
        g.link(resn, 2, cond, "length", "INT")

    # Pass-through, between conditioning and the sampler, so the report is
    # about the graph that is actually going to run.
    pre = g.add("MiniMaxH3Preflight", (-60, 640), size=(420, 260),
                inputs=[_in("conditioning", "CONDITIONING"),
                        _in("samples", "LATENT")],
                outputs=[_out("conditioning", "CONDITIONING"),
                         _out("samples", "LATENT"),
                         _out("sequence_length", "INT"),
                         _out("report", "STRING")],
                title="Preflight: what this render costs")
    g.link(cond, 0, pre, "conditioning", "CONDITIONING")
    g.link(cond, 1, pre, "samples", "LATENT")
    g.link(pre, 0, guider, "conditioning", "CONDITIONING")
    g.link(pre, 1, sampler, "latent_image", "LATENT")
    g.link(noise, 0, sampler, "noise", "NOISE")
    g.link(guider, 0, sampler, "guider", "GUIDER")
    g.link(samp, 0, sampler, "sampler", "SAMPLER")
    if not split_at:
        # With a split, SplitSigmas sits between these two and the link is
        # made below. This writer has no re-link, so a link made here would
        # be left dangling on the input it no longer owns.
        g.link(sched, 0, sampler, "sigmas", "SIGMAS")
    latent_src, latent_slot = sampler, 0

    if split_at:
        # See the matching note in build_api. ONE BasicScheduler feeds
        # SplitSigmas, so both halves sample the same curve -- that shared
        # schedule is the precondition, and it is why both stages must carry
        # the same shift.
        split = g.add("SplitSigmas", (400, 250), size=(300, 90),
                      widgets=[split_at],
                      inputs=[_in("sigmas", "SIGMAS")],
                      outputs=[_out("high_sigmas", "SIGMAS"),
                               _out("low_sigmas", "SIGMAS")],
                      title=f"Split the schedule at step {split_at}")
        g.link(sched, 0, split, "sigmas", "SIGMAS")
        g.link(split, 0, sampler, "sigmas", "SIGMAS")
        stage2_src = plain_src if split_base_last else model_src
        guider2 = g.add("BasicGuider", (400, 420), size=(300, 70),
                        inputs=[_in("model", "MODEL"), _in("conditioning", "CONDITIONING")],
                        outputs=[_out("GUIDER", "GUIDER")],
                        title="Stage 2 guider")
        g.link(stage2_src, 0, guider2, "model", "MODEL")
        g.link(pre, 0, guider2, "conditioning", "CONDITIONING")
        nonoise = g.add("DisableNoise", (400, 520), size=(300, 60),
                        outputs=[_out("NOISE", "NOISE")],
                        title="Stage 2 adds no noise")
        sampler2 = g.add("SamplerCustomAdvanced", (760, 250), size=(320, 150),
                         inputs=[_in("noise", "NOISE"), _in("guider", "GUIDER"),
                                 _in("sampler", "SAMPLER"), _in("sigmas", "SIGMAS"),
                                 _in("latent_image", "LATENT")],
                         outputs=[_out("output", "LATENT"),
                                  _out("denoised_output", "LATENT")],
                         title="Stage 2: finish")
        g.link(nonoise, 0, sampler2, "noise", "NOISE")
        g.link(guider2, 0, sampler2, "guider", "GUIDER")
        g.link(samp, 0, sampler2, "sampler", "SAMPLER")
        g.link(split, 1, sampler2, "sigmas", "SIGMAS")
        g.link(sampler, 0, sampler2, "latent_image", "LATENT")
        latent_src, latent_slot = sampler2, 0
    if stamp:
        # Bench only. Inline between the sampler and both decoders so it has a
        # real data dependency on the sampler -- ComfyUI orders by dependency,
        # not graph position, and without that edge it can legally run BEFORE
        # sampling and record pre-render state. SIGMAS is what makes n_sparse
        # computable; nothing else exposes it.
        stampn = g.add("MiniMaxH3ProvenanceStamp", (780, 240), size=(330, 130),
                       widgets=[f"bench {task}"],
                       inputs=[_in("latent", "LATENT"), _in("model", "MODEL"),
                               _in("sigmas", "SIGMAS", optional=True)],
                       outputs=[_out("latent", "LATENT")])
        g.link(sampler, 0, stampn, "latent", "LATENT")
        g.link(model_src, 0, stampn, "model", "MODEL")
        g.link(sched, 0, stampn, "sigmas", "SIGMAS")
        latent_src, latent_slot = stampn, 0
    # Link ORDER is preserved exactly as it was before the single-frame path
    # existed, including the two audio links sitting between the video decode
    # and the save. Link ids are assigned in call order, so reordering these
    # renumbers every link in all 24 UI graphs -- a 50-file diff that says
    # nothing, over a working tree other sessions are also editing.
    g.link(latent_src, latent_slot, vdec, "samples", "LATENT")
    g.link(vvae, 0, vdec, "vae", "VAE")
    if adec is not None:
        g.link(latent_src, latent_slot, adec, "samples", "LATENT")
        g.link(avae, 0, adec, "vae", "VAE")
    g.link(vdec, 0, save, "images", "IMAGE")
    if adec is not None:
        g.link(adec, 0, save, "audio", "AUDIO")

    # Guidance in the graph rather than in a doc nobody opens next to it.
    # MarkdownNote is in _UI_ONLY, so these never reach the API form and
    # cannot desync it.
    g.add("MarkdownNote", (-2180, 0), size=(620, 620), widgets=[_NOTE_GEOMETRY],
          title="Canvas + length: what is actually selectable")
    g.add("MarkdownNote", (-2180, 660), size=(620, 560), widgets=[_NOTE_NODES],
          title="Which nodes, and the order that matters")
    g.add("MarkdownNote", (-2860, 0), size=(640, 900), widgets=[_NOTE_SIZING],
          title="Resolution, references, and reading the preflight")
    if variant_note is not None:
        g.add("MarkdownNote", (-2180, 1280), size=(620, 760),
              widgets=[variant_note], title="What this graph is probing")

    return g.dump(title or f"h3-{task}-sage")


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
# of the combo list. Only `LoadImage.image` so far; add one when its node is
# read, not on the assumption that other loaders behave the same way.
_ANNOTATED_INPUTS = {("LoadImage", "image")}


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
            for option in (meta.get("options") or []):
                inner = (option.get("inputs") or {}) if isinstance(option, dict) else {}
                for section in ("required", "optional"):
                    for name in (inner.get(section) or {}):
                        known.setdefault(f"{parent}.{name}", None)

        given = node["inputs"]
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
    consumers = [(nid, n) for nid, n in graph.items()
                 if n["class_type"] in ("BasicScheduler", "BasicGuider")]
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


def validate_ui(wf: dict, oi: dict, label: str) -> list[str]:
    """Self-consistency only. No server validates a UI graph, so this checks
    what the frontend would choke on: dangling links and slot mismatches."""
    errs = []

    def e(msg):
        errs.append(f"{label}: {msg}")

    by_id = {n["id"]: n for n in wf["nodes"]}
    declared = {l[0] for l in wf["links"]}
    for lid, src, ss, dst, ds, t in wf["links"]:
        if src not in by_id or dst not in by_id:
            e(f"link {lid}: endpoint missing")
            continue
        s, d = by_id[src], by_id[dst]
        if ss >= len(s["outputs"]):
            e(f"link {lid}: output slot {ss} out of range on {s['type']}")
        elif lid not in (s["outputs"][ss]["links"] or []):
            e(f"link {lid}: not listed on {s['type']} output {ss}")
        if ds >= len(d["inputs"]):
            e(f"link {lid}: input slot {ds} out of range on {d['type']}")
        elif d["inputs"][ds].get("link") != lid:
            e(f"link {lid}: not recorded on {d['type']} input {ds}")
    for n in wf["nodes"]:
        # Frontend-only nodes have no backend class, so they are absent from
        # /object_info by design. Rejecting them would be the validator being
        # confidently wrong rather than the graph being broken.
        if n["type"] in _FRONTEND_ONLY:
            continue
        if n["type"] not in oi:
            e(f"node {n['id']}: unknown type {n['type']!r}")
            continue
        for i, inp in enumerate(n["inputs"]):
            if inp.get("link") is not None and inp["link"] not in declared:
                e(f"node {n['id']} ({n['type']}) input {inp['name']}: dangling link")
            if inp.get("link") is None and inp.get("shape") != 7 and "widget" not in inp:
                e(f"node {n['id']} ({n['type']}): required input {inp['name']} unconnected")
        # widgets_values must cover every widget the node declares, in order,
        # including any that have been converted to inputs.
        spec = oi[n["type"]]["input"]
        # `force_input=True` turns a scalar input into a socket, so it owns no
        # widget value. Missing that is how tau_profile got emitted as a 13th
        # widget on a 12-widget node: this check allows a surplus (see below),
        # so a spurious extra value was invisible from here.
        def _is_widget(v):
            opts = v[1] if len(v) > 1 and isinstance(v[1], dict) else {}
            if opts.get("forceInput"):
                return False
            return isinstance(v[0], list) or v[0] in (
                "INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3")

        widget_names = [k for k, v in ((spec.get("required") or {}) | (spec.get("optional") or {})).items()
                        if _is_widget(v)]
        got = len(n.get("widgets_values") or [])
        # RandomNoise / LoadImage carry an extra frontend-only widget
        # (control_after_generate, the upload button) that /object_info does
        # not report, so allow a surplus but never a shortfall.
        if got < len(widget_names):
            e(f"node {n['id']} ({n['type']}): {got} widget values for "
              f"{len(widget_names)} widgets {widget_names}")
    return errs


# --------------------------------------------------------------------------

# Nodes that are browser affordances rather than computation, so their
# absence from the API form is intentional and not drift.
#
# ModelPreviewOverrideKJ is the non-obvious one: it patches the model, but
# only to decode intermediate latents through taeh3 for display. Headless
# has nowhere to show them, and those decodes cost time that would land in
# any timing run as an unattributed confound. It belongs in the graph you
# watch and nowhere near the graph you measure.
#
# `PreviewImage` is kept in this set although nothing emits one: it is a stock
# node somebody may add to a UI graph by hand, and stripping it from the API
# form is right whether or not this generator produces it.
_UI_ONLY = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode",
            "ModelPreviewOverrideKJ", "PreviewImage"}

# Rendered entirely by the frontend, so they have no entry in /object_info.
# Subset of _UI_ONLY: ModelPreviewOverrideKJ is a real backend node that we
# exclude from the API form by choice, not by necessity.
_FRONTEND_ONLY = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode"}


def _ui_settings(wf):
    """{class_type: widgets} for a UI graph, ignoring bypassed nodes."""
    return {n["type"]: n.get("widgets_values")
            for n in wf["nodes"]
            if n["type"] not in _UI_ONLY and n.get("mode", 0) == 0}


def _api_settings(wf):
    """{class_type: non-link inputs} for an API graph."""
    return {n["class_type"]: {k: v for k, v in n["inputs"].items()
                              if not isinstance(v, list)}
            for n in wf.values()}


def cross_check(written):
    """Report where a task's UI and API graphs disagree.

    Compares which node counts are present and, for the ones carrying settings we
    pin, that the pinned values match. Widget *order* differs between the two
    formats by design (UI is positional, API is keyed), so this checks the
    node set plus the Sol-Attn and MiniMaxH3SageAttention values explicitly
    rather than trying to align every widget by index.
    """
    by_task = {}
    for task, fmt, p, wf in written:
        by_task.setdefault(task, {})[fmt] = (p.name, wf)

    errs = []
    for task, forms in sorted(by_task.items()):
        if len(forms) < 2:
            continue
        ui_name, ui = forms["ui"]
        api_name, api = forms["api"]
        ui_s, api_s = _ui_settings(ui), _api_settings(api)

        ui_counts = Counter(n["type"] for n in ui["nodes"]
                            if n["type"] not in _UI_ONLY
                            and n.get("mode", 0) == 0)
        api_counts = Counter(n["class_type"] for n in api.values())
        for cls in sorted(set(ui_counts) | set(api_counts)):
            if ui_counts[cls] != api_counts[cls]:
                errs.append(
                    f"{task}: {cls} count differs -- {ui_name} has "
                    f"{ui_counts[cls]}, {api_name} has {api_counts[cls]}")

        # Nodes whose values are compared, not just their presence. UI widgets
        # are positional in schema order; API inputs are keyed, so each entry
        # is the schema order of the widgets we care about.
        #
        # The Sol-Attn node is here because its settings have actually drifted.
        # UNETLoader and LoraLoaderModelOnly joined it the moment `unet` and
        # `lora` became free builder parameters: before that the checkpoint
        # was derived from `task` inside both builders and the two formats
        # could not disagree about it, and now they can. Which checkpoint a
        # graph loads is exactly the class of difference this function exists
        # to catch, and the node-set check above cannot see it -- both formats
        # carry a UNETLoader either way.
        #
        # The Sol node's widget order depends on its own `selection` value, so
        # it is read back out of the UI graph rather than assumed. Position 0
        # is the selector; the option's inputs follow it, and they are the
        # entries whose API key is dotted -- comparing `tau` against a keyed
        # `tau` would find nothing, because the API form no longer has one.
        sol_order = []
        sol_ui = ui_s.get(SOL_NODE)
        if sol_ui:
            selected = sol_ui[0]
            nested = SOL_SELECTION_INPUTS.get(selected)
            if nested is None:
                errs.append(f"{task}: {SOL_NODE} selection is {selected!r} in "
                            f"{ui_name}, which is not a declared option")
            else:
                sol_order = (["selection"] + [f"selection.{k}" for k in nested]
                             + list(SOL_TAIL_WIDGETS))

        for cls, order in (
            # Derived from the same tables the builder emits from rather than
            # repeated, so the drift check cannot itself drift from what the
            # builder emits -- a check comparing the generator to a stale copy
            # of the generator passes for the wrong reason.
            (SOL_NODE, sol_order),
            ("UNETLoader", ["unet_name"]),
            ("LoraLoaderModelOnly", ["lora_name", "strength_model"]),
            # The scheduler and step count joined on 2026-08-20, when a
            # scheduler other than `simple` first shipped in a graph. Nothing
            # else in the repo reads the scheduler, and a graph sampling the
            # wrong grid renders cleanly.
            ("BasicScheduler", ["scheduler", "steps"]),
            # The shifts are here for the same reason as the checkpoint: they
            # are a free builder value that the two formats can now disagree
            # about, and a graph sampling off the wrong schedule renders
            # cleanly rather than failing.
            ("MiniMaxH3SigmaShift", ["shift_video", "shift_audio"]),
        ):
            if cls not in ui_s or cls not in api_s:
                continue
            widgets = ui_s[cls] or []
            for i, key in enumerate(order):
                if i >= len(widgets) or key not in api_s[cls]:
                    continue
                if widgets[i] != api_s[cls][key]:
                    errs.append(
                        f"{task}: {cls}.{key} is {widgets[i]!r} in "
                        f"{ui_name} but {api_s[cls][key]!r} in {api_name}")

        # VAELoader is compared as a SET of filenames rather than through the
        # keyed dicts above, and it has to be: every graph loads two VAEs, and
        # `_ui_settings`/`_api_settings` key by CLASS NAME, so the second
        # VAELoader silently overwrites the first and whichever survives is an
        # accident of iteration order. Adding "VAELoader" to the list above
        # would compare one arbitrary loader against another.
        #
        # It is here for the same reason UNETLoader is: `vae_name` became a
        # free builder value when the single-frame path introduced the image
        # VAE, so the two formats can now disagree about which decoder a graph
        # loads. That difference renders cleanly and looks wrong only in the
        # pixels -- a video graph on the one-frame VAE, or the reverse.
        ui_vaes = sorted(
            str((n.get("widgets_values") or [None])[0]) for n in ui["nodes"]
            if n["type"] == "VAELoader" and n.get("mode", 0) == 0)
        api_vaes = sorted(str(n["inputs"].get("vae_name")) for n in api.values()
                          if isinstance(n, dict) and n.get("class_type") == "VAELoader")
        if ui_vaes != api_vaes:
            errs.append(f"{task}: the two forms load different VAEs -- "
                        f"{ui_name} has {ui_vaes}, {api_name} has {api_vaes}")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object-info", default="http://127.0.0.1:8188",
                    help="running ComfyUI base URL, or a path to a saved object_info.json")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--no-validate", action="store_true")
    # Loading the right prompt into the right arm, without opening a JSON.
    # The graphs already ship with theirs baked in; these are for pasting one
    # into a graph you are editing by hand, or reading one without ComfyUI.
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
    # the task abbreviation the code uses internally. All carry the taeh3
    # preview, which is what lets a bad seed die at ~90s instead of costing a
    # full render -- worth more than any kernel knob when render time is the
    # objective.
    # `label` keys the UI/API cross-check and has to be unique; `task` is what
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
        ("h3_image_ref_plus_text_to_video.json", "r2v", "r2v", _ref_prompt(images=True), {},
         "reference image(s) + text -> video + audio"),
        ("h3_first_frame_to_video.json", "i2v", "i2v", None, {},
         "first frame + text -> video + audio (via MiniMaxH3KeyframeCanvas)"),

        # fl2va: the same node, the same task string, one more LoadImage.
        # `last_frame` is what separates them, which is why the task stays
        # "i2v" -- inventing a fourth task value would fork the geometry and
        # canvas logic that both modes share exactly.
        ("h3_first_last_frame_to_video.json", "fl2v", "i2v", None,
         dict(last_frame=True, variant_note=_NOTE_FL2V,
              out_prefix="Video/h3_fl2v", **FL2V_CANVAS),
         "first frame + last frame + text -> video + audio"),

        # The first turbo graph here that is IN distribution: every released
        # turbo LoRA is an fl2v distill and every previous turbo arm was t2v or
        # ref2v. Vendor row for the 4-step 768p, cloned from
        # h3_text_to_video_turbo_4step_768p rather than retyped.
        ("h3_first_last_frame_to_video_turbo_4step_768p.json", "fl2v-turbo768",
         "i2v", None,
         dict(last_frame=True,
              lora=(TURBO_768P_LORA, TURBO_768P_STRENGTH),
              steps=TURBO_768P_STEPS, shift=TURBO_768P_SHIFT,
              sampler_name=TURBO_SAMPLER,
              variant_note=_NOTE_FL2V_TURBO,
              out_prefix="Video/h3_fl2v_turbo_4step_768p", **FL2V_CANVAS),
         "first + last frame + text -> video + audio, via the 4-step 768p "
         "turbo LoRA at shift 6 -- in distribution"),
        # t2v deliberately: the note explains that matching the LoRA's 544p
        # means leaving H3's own canvas rule, and MiniMaxH3KeyframeCanvas is
        # the node that refuses to, so an i2v turbo graph could not show the
        # choice it is describing.
        ("h3_text_to_video_turbo.json", "t2v-turbo", "t2v", LONG_T2V_PROMPT,
         dict(lora=(TURBO_LORA, TURBO_LORA_STRENGTH), steps=TURBO_STEPS,
              shift=TURBO_SHIFT, variant_note=_NOTE_TURBO,
              out_prefix="Video/h3_t2v_turbo_8step"),
         "text -> video + audio, via the 8-step turbo LoRA"),

        # The 4-step 768p LoRA: one of the two released turbos whose shift
        # is not 12/3. It exists as a shipped graph rather than as a note because
        # "change the shift when you change the LoRA" is the instruction
        # everyone drops, and a graph that already has it right is worth more
        # than a paragraph saying to do it. Its training canvas IS the default
        # canvas, so unlike the 8-step nothing else has to move.
        ("h3_text_to_video_turbo_4step_768p.json", "t2v-turbo768", "t2v",
         LONG_T2V_PROMPT,
         dict(lora=(TURBO_768P_LORA, TURBO_768P_STRENGTH),
              steps=TURBO_768P_STEPS, shift=TURBO_768P_SHIFT,
              variant_note=_NOTE_TURBO_768P,
              out_prefix="Video/h3_t2v_turbo_4step_768p"),
         "text -> video + audio, via the 4-step 768p turbo LoRA at shift 6"),

        # The recipe the 2026-08-20 blind session supports: the vendor row with
        # the vendor's sampler. Differs from h3_text_to_video_turbo_4step_768p
        # in one widget (er_sde -> euler); differs from the owner graph below in
        # scheduler and strength, which the session found indistinguishable at
        # 20% more sampler time. Ships whatever TURBO_768P_LORA names, which
        # has been v1.1 since 2026-08-23 -- this comment said it ships v1.0
        # "because only v1.0 has an attested row", which was the argument for
        # not adopting v1.1 and is no longer the state.  The row is now
        # inherited rather than attested; see check_distill_settings.UNATTESTED.
        ("h3_text_to_video_turbo_768p_euler.json", "t2v-turbo768-euler", "t2v",
         LONG_T2V_PROMPT,
         dict(lora=(TURBO_768P_LORA, TURBO_768P_STRENGTH),
              steps=TURBO_768P_STEPS, shift=TURBO_768P_SHIFT,
              sampler_name=TURBO_SAMPLER,
              out_prefix="Video/h3_t2v_turbo_768p_euler",
              variant_note=_probe_note(
                  "the recipe the blind session supports",
                  "h3_text_to_video_turbo_4step_768p.json",
                  f"the sampler: `{TURBO_SAMPLER}` instead of "
                  f"`{SAMPLING['sampler']}`, the vendor's own choice for its "
                  "distilled LoRAs. Everything else is the vendor row: 4 steps, "
                  "shift 6/3, `simple`, strength 1.0, Sol on at the shipped tau.",
                  "Nothing in particular; this is a working graph, not an "
                  "experiment. It exists so the recipe has a sha.",
                  "The same clip family as its twin. On 2026-08-20 the owner "
                  "could not tell this recipe from euler/beta/0.75 across 8 "
                  "seeds, and it is 20% faster on the card.")),
         "text -> video + audio: the 768p turbo LoRA at the vendor row with euler"),

        # The owner's working recipe on the same LoRA, as a graph with a sha
        # so bench arms can patch the LoRA file onto it. Three knobs differ
        # from the row above; see TURBO_OWNER_STRENGTH in h3_config.
        ("h3_probe_turbo_768p_owner.json", "t2v-turbo768-owner", "t2v",
         LONG_T2V_PROMPT,
         dict(lora=(TURBO_768P_LORA, TURBO_OWNER_STRENGTH),
              steps=TURBO_768P_STEPS, shift=TURBO_768P_SHIFT,
              sampler_name=TURBO_SAMPLER, scheduler_name=TURBO_OWNER_SCHEDULER,
              out_prefix="Video/h3_probe_turbo_768p_owner",
              variant_note=_NOTE_TURBO_OWNER),
         f"the 768p turbo LoRA at the owner's recipe: euler, beta, "
         f"{TURBO_768P_STEPS} steps, strength {TURBO_768P_STRENGTH:g}"),

        # The SLA release on the same row. A probe rather than a shipped
        # variant because nothing is known about how a LoRA distilled under
        # a top-k block router behaves under Sol's threshold router, and the
        # first render is the first datum. See h3_config.TURBO_SLA_LORA.
        ("h3_probe_turbo_768p_sla.json", "t2v-turbo768-sla", "t2v",
         LONG_T2V_PROMPT,
         dict(lora=(TURBO_SLA_LORA, TURBO_LORA_STRENGTH),
              steps=TURBO_SLA_STEPS, shift=TURBO_SLA_SHIFT,
              out_prefix="Video/h3_probe_turbo_768p_sla",
              variant_note=_probe_note(
                  "whether a LoRA distilled under sparse attention survives a "
                  "different sparse attention",
                  "h3_text_to_video_turbo_4step_768p.json",
                  "the LoRA file: lightx2v's Turbo-SLA 4-step v0.1 768p instead "
                  f"of the Turbo {turbo_label(TURBO_768P_LORA)}. Same shift (6/3), same steps, "
                  "same rank, alpha, base and tensor keys -- the file differs in "
                  "what the student saw during distillation. The SLA student's "
                  "attention ran a top-k block router that keeps 15% of key "
                  "blocks per query block (reported from the model card and "
                  "LightX2V's config for it; the training code is not in any "
                  "checkout here). This graph runs it under Sol-Attn, a "
                  "threshold router with a dense fallback, because that is "
                  "the repo default and the only sparse kernel on this box.",
                  "Whether it renders a coherent clip at all, first. Then the "
                  "same things as any turbo arm: motion, texture, audio. There "
                  "is no kernel here that reproduces the router it was trained "
                  "under, so this is not a test of SLA -- it is a test of "
                  "whether SLA's LoRA transfers to a router it never saw.",
                  "Unknown in both directions. The SLA paper's claim is that a "
                  "fine-tuned model under its sparse router matches the dense "
                  "original; it says nothing about that model under another "
                  "router or under dense attention. If this arm degrades "
                  "relative to its twin, the candidate cause is the router "
                  "mismatch and the control is the same pair with Sol "
                  "bypassed, which nothing here has rendered either.")),
         "the 768p turbo graph with lightx2v's SLA-distilled LoRA swapped in"),

        # The same SLA arm under the router it was distilled with, and under
        # no sparse attention at all. With the probe above (Sol) these are the
        # three attention regimes; bench arms patch the v1.1 file onto all
        # three for the never-SLA-trained control. Everything but the
        # attention chain is the SLA probe's.
        ("h3_probe_turbo_768p_sla_router.json", "t2v-turbo768-sla-router", "t2v",
         LONG_T2V_PROMPT,
         dict(lora=(TURBO_SLA_LORA, TURBO_LORA_STRENGTH),
              steps=TURBO_SLA_STEPS, shift=TURBO_SLA_SHIFT,
              sla_router=0.85,
              out_prefix="Video/h3_probe_turbo_768p_sla_router",
              variant_note=_probe_note(
                  "whether the SLA LoRA behaves differently under the "
                  "attention it was distilled with",
                  "h3_probe_turbo_768p_sla.json",
                  "the attention chain: `MiniMaxH3SLARouter` at sparsity 0.85 "
                  "(LightX2V's sparse top-k block router on its Triton kernel, "
                  "vendored at `vendor/sla_sparse_triton.py`, gated on captured "
                  "activations in `bench/results/2026-08-20_sla_router_gate.json`) "
                  "instead of sage + Sol-Attn. No sage node, no Sol node; the "
                  "chain assert runs warn-only. Same LoRA, steps, shift, seed, "
                  "prompt. Two named gaps from the release's own path: 64/64 "
                  "blocks against its 128/64, and bf16 Triton against "
                  "SpargeAttn's sage2 operator.",
                  "Whether it renders a coherent clip, and what the sampler "
                  "costs against the Sol twin -- the router keeps 15% of key "
                  "blocks where Sol at tau 1.0 keeps more, so this is also the "
                  "speed question. Quality is briefs met, one seed.",
                  "Unknown. On the base model's activations the router's error "
                  "at 0.85 was comparable to Sol's at tau 1.0-1.3 on ordinary "
                  "heads and better on the loudest (the gate record); the "
                  "student was trained to survive this cut, so its own "
                  "activations may do better still. The captures under the "
                  "LoRA are what settle that.")),
         "the SLA LoRA under the sparse top-k router it was distilled with"),

        ("h3_probe_turbo_768p_sla_dense.json", "t2v-turbo768-sla-dense", "t2v",
         LONG_T2V_PROMPT,
         dict(lora=(TURBO_SLA_LORA, TURBO_LORA_STRENGTH),
              steps=TURBO_SLA_STEPS, shift=TURBO_SLA_SHIFT,
              sol_on=False,
              out_prefix="Video/h3_probe_turbo_768p_sla_dense",
              variant_note=_probe_note(
                  "whether the SLA LoRA needs sparse attention at all",
                  "h3_probe_turbo_768p_sla.json",
                  "Sol-Attn absent: sage only, the repo's dense-baseline "
                  "convention (`workflows/bench/*_stamped_api.json`). Every "
                  "block the student learned to do without is back.",
                  "Coherence and sampler cost against the Sol and router twins.",
                  "Unknown; the SLA paper's claim is about the fine-tuned "
                  "model under its sparse router, not under dense attention. "
                  "Not a discharge of `docs/open_experiments.md` #9, which is "
                  "stock torch attention with no sage either.")),
         "the SLA LoRA with Sol-Attn absent: sage only"),

        # First graph in this repo to wire a reference VIDEO. Everything about
        # that path was read off source until 2026-08-13 and never executed.
        # The reference-combination matrix. Five arms, one per shape of
        # ref2va request, each with a prompt that declares EXACTLY the labels
        # its own graph wires -- `bench/check_ref_prompt_labels.py` enforces
        # that agreement, because the tokenizer derives the labels from the
        # sockets and a prompt naming one that is not there fails silently.
        ("h3_ref_video_to_video.json", "r2v-video", "r2v",
         _ref_prompt(images=True, video=True, video_audio=True),
         dict(**REF_VIDEO_BUDGET, ref_video=True, out_prefix="Video/h3_r2v_video",
              variant_note=_NOTE_REF_VIDEO),
         "images + reference video + its soundtrack -> video + audio"),

        ("h3_ref_video_only.json", "r2v-video-only", "r2v",
         _ref_prompt(images=False, video=True),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_video_audio=False, ref_images_on=False,
              out_prefix="Video/h3_r2v_video_only",
              variant_note=_note_ref_matrix("a reference video and nothing else")),
         "reference video only, silent clip"),

        ("h3_ref_video_audio.json", "r2v-video-audio", "r2v",
         _ref_prompt(images=False, video=True, video_audio=True),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_images_on=False,
              out_prefix="Video/h3_r2v_video_audio",
              variant_note=_note_ref_matrix("a reference video with its own soundtrack")),
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
              out_prefix="Video/h3_probe_release_video_policy",
              variant_note=_probe_note(
                  "what release-matched two-stage reference-video preparation changes",
                  "h3_ref_video_audio.json",
                  "only `video_policy`: release instead of comfy. The local "
                  "compiler upscales the full-rate VAE view to the release "
                  "canvas and independently runs the raw 2 fps Qwen samples "
                  "through the release's duration-aware processor.",
                  "Preflight's VAE/Qwen geometry block and the `[h3] reference "
                  "video ... policy=release` server line before judging the clip.",
                  "More VAE reference rows than the comfy twin, while Qwen "
                  "lands on its own duration-budgeted grid rather than blindly "
                  "sharing the upscaled frames. This is an opt-in local parity "
                  "policy over native-open ComfyUI gaps, not an upstream fix.")),
         "reference video on the atomic release VAE/Qwen preparation policy"),

        ("h3_ref_image_audio.json", "r2v-image-audio", "r2v",
         _ref_prompt(images=True, audio=True),
         dict(ref_audio=True, out_prefix="Video/h3_r2v_image_audio",
              variant_note=_note_ref_matrix("reference images and a standalone audio clip")),
         "reference images + standalone audio"),

        ("h3_ref_image_video_audio.json", "r2v-all", "r2v",
         _ref_prompt(images=True, video=True, video_audio=True, audio=True),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_audio=True,
              out_prefix="Video/h3_r2v_all",
              variant_note=_note_ref_matrix("every reference type at once")),
         "images + video + its soundtrack + standalone audio"),

        # A CAPTURE target, not a render target. It exists so `h3_capture.py`
        # has a graph to point at, and so the next capture is a re-run rather
        # than archaeology: the 2026-08-15 capture's config survives only as
        # prose in a README on the share, which is why nothing about it could
        # be reproduced from this repo.
        #
        # Sol-Attn is absent, as in every UI graph, and that is a REQUIREMENT
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
         # sol_on=False, and this is the ONE graph that earns the exception to
         # the Sol-on-by-default rule: `h3_capture.py` records the activations a
         # dense baseline is measured from, and a Sol arm gives sage only a
         # subset of the sampler's steps, so a capture taken through Sol is a
         # different trajectory than the one the analysis assumes.
         #
         # It was previously done by hand-editing the emitted `_api.json` to
         # route past the Sol node, which CLAUDE.md forbids and which the next
         # regeneration silently reverted -- putting Sol back into the capture
         # chain with nothing going red. Declaring it here survives regeneration
         # and fixes the UI twin in the same move.
         dict(**REF_VIDEO_BUDGET, ref_images=CAPTURE_REF_IMAGES,
              sol_on=False,
              out_prefix="Video/h3_probe_capture_ref3"),
         "3 references spanning 0.78-4.23 MP; the h3_capture.py target"),

        # The VAE encoder-precision arms. ComfyUI carries ONE dtype per VAE and
        # `--fp32-vae` moves both halves; the release keeps this VAE resident in
        # fp32 *because the same module encodes references and keyframes* and
        # decodes under fp16, which the flag cannot express.
        # `MiniMaxH3VAEPrecision` splits them, and until 2026-08-21 it was wired
        # into zero graphs -- the `MiniMaxH3VendorTokens` failure mode, a fix
        # nobody can reach.
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

        # Reference transfer of the fl2v distill, four checkpoints, one
        # variable. The LoRA file is patched at run time; see the note.
        *[
            (f"h3_probe_ref_turbo768p_{tag}.json", f"r2v-turbo768-{tag}", "r2v",
             _ref_prompt(images=("character", "garment", "environment")),
             dict(**REF_VIDEO_BUDGET, ref_images=CAPTURE_REF_IMAGES,
                  unet=MODELS[key],
                  lora=(TURBO_768P_LORA, TURBO_768P_STRENGTH),
                  steps=TURBO_768P_STEPS, shift=TURBO_768P_SHIFT,
                  sampler_name=TURBO_SAMPLER,
                  out_prefix=f"Video/h3_probe_ref_turbo768p_{tag}",
                  variant_note=_note_ref_transfer(label, what)),
             f"the capture request on {label} with the 4-step 768p turbo LoRA")
            for tag, key, label, what in (
                ("fl2va", "unet_fl2va", "fl2va",
                 "The checkpoint the LoRA was distilled on, and one that never "
                 "saw a reference row."),
                ("hybrid_b30", "unet_hybrid_b30", "the HF hybrid b30-49",
                 "fl2va's linears with ref2va's modulation in the last twenty "
                 "blocks."),
                ("hybrid_adaln_all", "unet_hybrid_adaln_all",
                 "the locally built all-adaln hybrid",
                 "fl2va's linears with ref2va's modulation in every block and "
                 "the final layer -- the adaln-only hypothesis as a file."),
                ("ref2va", "unet_ref2va", "ref2va",
                 "The checkpoint the task belongs to, with linears the LoRA "
                 "was not fitted against."),
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
              sol_on=False, unet=MODELS["unet_fl2va"],
              out_prefix="Video/h3_probe_capture_ref3_fl2va"),
         "the capture twin on fl2va with no LoRA; the missing block-49 control"),

        # Gate 6, the reference-view ablation: three ref2va arms from one base
        # graph (the capture request, three stills spanning 0.78-4.23 MP at
        # the vendor row), differing only in how each still reaches its two
        # consumers. `docs/h3_conditioning_end_to_end.md` section 1b is why the
        # VAE view and the Qwen view need not share a geometry;
        # `docs/h3_references.md` prices the two stages. The encoder is the
        # ComfyUI-native INT8 ConvRot file BY NAME (`ENCODER_INT8`), the
        # encoder of record since the four-encoder holdout table of
        # 2026-08-25; the W4 artifacts (`MODELS["clip"]`, `ENCODER_V2`) swap in
        # at the combo without a graph edit for the small-host variant. The
        # native file takes the 2048 Qwen view directly (core's own sizing),
        # where the v1 artifact clamps. `video_policy=release` is set with no video reference wired,
        # so a video row patched in through `run_graph_arms --set` inherits the
        # release sizing rather than a policy nobody chose. Sol on, as in every
        # shipped video graph. `bench/gate6_refview_arms.json` is the manifest
        # `run_graph_arms.py --manifest` consumes for matched-seed pairs.
        *[
            (f"h3_probe_refview_{tag}.json", f"r2v-refview-{tag}", "r2v",
             _ref_prompt(images=("character", "garment", "environment")),
             dict(**{**REF_VIDEO_BUDGET, "ref_upscale": upscale},
                  ref_images=CAPTURE_REF_IMAGES, clip=ENCODER_INT8,
                  ref_video_policy="release",
                  ref_qwen_short_edge=qwen,
                  out_prefix=f"Video/h3_probe_refview_{tag}",
                  variant_note=note),
             what)
            for tag, upscale, qwen, what, note in (
                ("a_source", False, 0,
                 "arm A: every still at source size for both consumers, no upscale",
                 "Reference-view ablation, arm A. Stage one leaves each still at "
                 "its source size (allow_upscale off) and one view feeds both the "
                 "video VAE and Qwen3-VL. The no-upscale baseline the other two "
                 "arms are judged against, blind, as a distribution of seeds."),
                ("b_qwen2048", False, _ref_short_edge(),
                 "arm B: the Qwen view alone scaled to the vendor short edge; the VAE keeps the source",
                 "Reference-view ablation, arm B. The VAE view is arm A's; the "
                 "Qwen view alone is rescaled so its shorter side reaches the "
                 "vendor short edge (`qwen_short_edge`, one Lanczos resample). "
                 "Answers whether the encoder wants the upscale when the VAE "
                 "does not pay for it. Under the v1 snapshot the encoder's own "
                 "bounds clamp this view back; the v2 artifact admits it."),
                ("c_parity", True, 0,
                 "arm C: full vendor parity, both consumers at the upscaled view where the canvas allows",
                 "Reference-view ablation, arm C. allow_upscale on: stage one "
                 "scales each still toward the vendor short edge as the canvas "
                 "allows and the same view feeds the VAE and Qwen3-VL, which is "
                 "the release pipeline's geometry and the reference-latent rows "
                 "it costs. Priced by preflight; judged against A and B blind."),
            )
        ],

        ("h3_ref_video_edit.json", "r2v-edit", "r2v",
         _ref_prompt(images=False, video=True, video_audio=True, video_role="edit"),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_images_on=False,
              out_prefix="Video/h3_r2v_edit",
              variant_note=_note_ref_relationship("edit")),
         "edit a source video -- the closest thing H3 has to inpainting"),

        ("h3_ref_video_image_edit.json", "r2v-edit-combo", "r2v",
         _ref_prompt(images=True, video=True, video_audio=True, video_role="edit"),
         dict(**REF_VIDEO_BUDGET, ref_video=True,
              out_prefix="Video/h3_r2v_edit_combo",
              variant_note=_note_ref_relationship("edit")),
         "edit a source video, with images supplying what replaces what"),

        # The twin of h3_ref_video_image_edit: same sockets, same budget, a
        # different request. Kept adjacent so the pair reads as the A/B it is.
        ("h3_ref_video_swap.json", "r2v-swap", "r2v",
         _ref_prompt(images=True, video=True, video_audio=True,
                     video_role="swap", audio_role="copy"),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_image_count=1,
              out_prefix="Video/h3_r2v_swap",
              variant_note=_note_ref_relationship("swap")),
         "replace a character in a source video with one from an image"),

        ("h3_ref_video_continue.json", "r2v-continue", "r2v",
         _ref_prompt(images=False, video=True, video_audio=True, video_role="continue"),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_images_on=False,
              out_prefix="Video/h3_r2v_continue",
              variant_note=_note_ref_relationship("continue")),
         "continue from the end of a source video"),

        ("h3_ref_video_motion.json", "r2v-motion", "r2v",
         _ref_prompt(images=True, video=True, video_role="motion"),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_video_audio=False,
              out_prefix="Video/h3_r2v_motion",
              variant_note=_note_ref_relationship("motion")),
         "transfer motion from a video onto a subject from an image"),

        ("h3_ref_audio_voice.json", "r2v-voice", "r2v",
         _ref_prompt(images=True, audio=True, audio_role="voice"),
         dict(ref_audio=True, out_prefix="Video/h3_r2v_voice",
              variant_note=_note_ref_relationship("voice")),
         "reference a speaker's voice timbre for generated speech"),

        # --- probes: pairs, one variable, run against the named twin ---

        ("h3_probe_split_base_last.json", "t2v-split-baselast", "t2v",
         LONG_T2V_PROMPT,
         dict(lora=(TURBO_LORA, TURBO_LORA_STRENGTH), steps=TURBO_STEPS,
              shift=TURBO_SHIFT, split_at=SPLIT_AT, split_base_last=True,
              out_prefix="Video/h3_probe_split_baselast",
              variant_note=_note_split(True)),
         "distilled high-noise, plain base model finishes"),

        ("h3_probe_split_base_first.json", "t2v-split-basefirst", "t2v",
         LONG_T2V_PROMPT,
         dict(lora=(TURBO_LORA, TURBO_LORA_STRENGTH), steps=TURBO_STEPS,
              shift=TURBO_SHIFT, split_at=SPLIT_AT, split_base_last=False,
              out_prefix="Video/h3_probe_split_basefirst",
              variant_note=_note_split(False)),
         "plain base high-noise, distilled finish (the Krea 2 ordering)"),

        ("h3_probe_turbo_home_canvas.json", "t2v-turbo-544p", "t2v",
         LONG_T2V_PROMPT,
         dict(lora=(TURBO_LORA, TURBO_LORA_STRENGTH), steps=TURBO_STEPS,
              shift=TURBO_SHIFT, **TURBO_HOME_CANVAS,
              out_prefix="Video/h3_probe_turbo_544p",
              variant_note=_probe_note(
                  "whether a 544p LoRA would rather have its own canvas",
                  "h3_text_to_video_turbo.json",
                  "960x544 instead of 1344x768. Same LoRA, same steps, same "
                  "shift, same seed and prompt -- only the canvas moved, onto "
                  "the resolution the 8-step v1.0 was actually distilled at.",
                  "Whether the output is better, not whether it is faster. It "
                  "will be faster: 510 tokens/frame against 1008, i.e. 0.26x "
                  "the attention. That is not the question.",
                  "Unknown, and that is the point. You cannot satisfy both "
                  "distributions at once: at 1344x768 the base model is home "
                  "and the LoRA is stretched to roughly twice the sequence it "
                  "was distilled on; at 960x544 the LoRA is home and the base "
                  "model is below H3's own 768 short edge, outside the canvas "
                  "family it was trained on. The vendor's own graph ships "
                  "960x544, which is their answer, not a measurement.")),
         "the 8-step turbo LoRA at the 544p it was distilled at"),

        ("h3_probe_turbo_euler.json", "t2v-turbo-euler", "t2v", LONG_T2V_PROMPT,
         dict(lora=(TURBO_LORA, TURBO_LORA_STRENGTH), steps=TURBO_STEPS,
              shift=TURBO_SHIFT, sampler_name=TURBO_SAMPLER,
              out_prefix="Video/h3_probe_turbo_euler",
              variant_note=_probe_note(
                  "whether a distilled model wants a first-order sampler",
                  "h3_text_to_video_turbo.json",
                  f"sampler `{TURBO_SAMPLER}` instead of "
                  f"`{SAMPLING['sampler']}`. The scheduler stays `simple`, "
                  "which is not a free choice: `simple` reproduces the "
                  "distillation's own sigma grid EXACTLY at every shift and "
                  "step count, and every other scheduler deviates from it.",
                  "Prompt adherence and motion, not speed. Both samplers are "
                  "one model eval per step, so this costs nothing either way.",
                  "The vendor ships euler on both their turbo graphs while "
                  "core ships res_multistep on the base ones, which reads as "
                  "deliberate. The argument: a distilled model is trained so "
                  "ONE Euler step from sigma_i lands at sigma_i+1, so a "
                  "multistep integrator corrects a discretization error that "
                  "is not the dominant error here, and perturbs a trajectory "
                  "that was already trained to be right. That is an argument, "
                  "not a measurement, which is why this is a pair.")),
         "the turbo graph with the vendor's sampler"),

        # The equal-cost shape control. 21:9, 16:9 and 9:16 are all
        # (w//32)*(h//32) = 1008 tokens/frame, so all three run at the SAME
        # sequence length and the same attention cost while the long edge goes
        # 768 -> 1536. Every other probe here changes cost to change shape;
        # these two change shape with cost held exactly constant, which is the
        # only way to ask whether the model is actually shape-neutral.
        ("h3_probe_canvas_ultrawide.json", "t2v-21by9", "t2v", LONG_T2V_PROMPT,
         dict(width=1536, height=672, out_prefix="Video/h3_probe_21by9",
              variant_note=_probe_note(
                  "shape at constant cost, the long way",
                  "h3_text_to_video.json",
                  "1536x672 instead of 1344x768. Both are 1008 tokens/frame, "
                  "so the sequence length, the attention cost and the render "
                  "time are the same by construction. The long edge went from "
                  "1344 to 1536. **1536 is not the end of that axis**: the "
                  "legal 1:4..4:1 family holds eight canvases at exactly 1008 "
                  "tokens/frame -- 1344x768, 1536x672, 1792x576 and 2016x512, "
                  "plus each of those transposed -- so the equal-cost run goes "
                  "to a 3.94:1 frame. This probe takes one step along it, not "
                  "the last one.",
                  "Composition and coherence across the wide axis, not speed. "
                  "Preflight's sequence length should be IDENTICAL to the "
                  "twin's -- if it is not, one of the two canvases is not "
                  "what this note claims.",
                  "Unknown. Every number in this repo was taken at 16:9, so "
                  "whether the model handles a 2.29:1 frame as well as a "
                  "1.75:1 one has never been asked. Cost cannot explain any "
                  "difference you see, which is what makes this worth "
                  "running.")),
         "21:9, the same cost as the default canvas"),

        ("h3_probe_canvas_portrait.json", "t2v-9by16", "t2v", LONG_T2V_PROMPT,
         dict(width=768, height=1344, out_prefix="Video/h3_probe_9by16",
              variant_note=_probe_note(
                  "shape at constant cost, the tall way",
                  "h3_text_to_video.json",
                  "768x1344 instead of 1344x768. Packed rows are "
                  "(w//32)*(h//32), which is symmetric, so portrait and "
                  "landscape of a ratio cost exactly the same: 1008 "
                  "tokens/frame either way.",
                  "Whether the model is orientation-neutral. 16:9 against "
                  "9:16 is a quality question here, never a speed one.",
                  "Unknown, and the symmetry is the point: if portrait looks "
                  "worse it is the training distribution talking, not the "
                  "geometry. Run this against the ultrawide probe and the "
                  "default and you have three shapes at one price.")),
         "9:16 portrait, the same cost as the default canvas"),

        ("h3_probe_ref2v_turbo.json", "r2v-turbo", "r2v", _ref_prompt(images=True),
         dict(lora=(TURBO_LORA, TURBO_LORA_STRENGTH), steps=TURBO_STEPS,
              shift=TURBO_SHIFT,
              out_prefix="Video/h3_probe_r2v_turbo",
              variant_note=_NOTE_REF2V_TURBO),
         "ref2v with an fl2v turbo LoRA -- deliberately out of distribution"),
        # The twin of the arm above, and the only difference that matters is
        # WHICH turbo LoRA. That one is an fl2v distill touching 208 modules,
        # none of them the conditioning-modulation path. This one touches 259,
        # the extra 51 being every `adaln_proj.linear` including
        # `final_layer`'s -- exactly where fl2va and ref2va diverge most.
        # See docs/h3_ref2v_distillation.md for the header measurement.
        #
        # Its own README claims t2v and i2v only and never mentions ref2va, so
        # this arm is OUR experiment, not the author's claim. Settings are the
        # pack's own; the graph differs from its twin in the two nodes the
        # pack requires, not in shift, canvas, seed or prompt.

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
        # EULER, not the repo's er_sde default. This is replication, not
        # preference: alibaba-pai's own scheduler takes "one Euler (eta = 0)
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

        # The arm PDD is actually claiming to beat, matched to
        # h3_image_ref_plus_text_to_video_pdd_4step on every axis a comparison
        # needs: same canvas, length, prompt, seed, sampler, scheduler, shift
        # and step count. The LoRA is the only thing that differs.
        #
        # Both are ref2v-NATIVE distills, which matters: docs/h3_ref2v_distillation.md
        # is about fl2v turbos being aimed at the wrong weight partition, and
        # neither of these is. So this pair asks about the METHOD rather than
        # about partition mismatch.
        #
        # sage on and Sol absent, matching its twin. Sol skips attention
        # adaptively per step and that is incoherent against a 4-step schedule
        # for either distill, so leaving it in would vary attention as well as
        # the LoRA.
        ("h3_image_ref_plus_text_to_video_turbo_4step.json", "r2v-turbo4", "r2v",
         _ref_prompt(images=True),
         dict(sampler_name="euler",
              lora=(TURBO_REF2VA_LORA, 1.0), steps=TURBO_REF2VA_STEPS,
              shift=TURBO_REF2VA_SHIFT,
              out_prefix="Video/h3_r2v_turbo_4step",
              variant_note=_probe_note(
                  "does PDD beat the turbo distill it is pitched against",
                  "h3_image_ref_plus_text_to_video_pdd_4step.json",
                  "the lightx2v ref2v turbo instead of the PDD LoRA, at the "
                  "same 4 evaluations and the same 12/3 shift.",
                  "identity on the reference subject and texture late in the "
                  "clip, which is where PDD's fused heads differ most from the "
                  "base and where the turbo LoRAs touch nothing.",
                  "PDD perturbs the backbone about 20x harder than the 8-step "
                  "turbo and moves the modulation path the turbos leave "
                  "alone; this is where that shows or does not. NOTE the "
                  "turbo was distilled at 544p and this renders 768p.")),
         "the ref2v turbo at 4 steps, matched to the PDD 4-step arm"),

        # --- the dialogue probe, t2v and ref2v -----------------------------
        # Eight short lines over three shots with the pacing written in. The
        # pair exists because they differ on exactly one axis: where the two
        # people come from. Everything judgeable -- the lines, the cut times,
        # the speaker order, the soundscape -- is identical, so anything that
        # separates them is the reference conditioning and not the script.
        #
        # 362 frames, which is the longest length H3 was trained on and the
        # last shot needs it: shot 3 starts at 00:11.000.
        ("h3_text_to_video_dialogue.json", "t2v-dialogue", "t2v",
         DIALOGUE_T2V_PROMPT,
         dict(length=362, out_prefix="Video/h3_t2v_dialogue",
              variant_note=_probe_note(
                  "does the dialogue marker survive at speed",
                  "h3_image_ref_plus_text_to_video_dialogue.json",
                  "no references: both speakers are described in prose.",
                  "whether either voice speaks the marker or the word "
                  "English aloud, and whether the seven turn boundaries land "
                  "on the right mouth.",
                  "the ref2v twin runs the same eight lines from two stills, "
                  "so the two are readable against each other.")),
         "two speakers, eight clipped lines, dialogue markers throughout"),

        ("h3_image_ref_plus_text_to_video_dialogue.json", "r2v-dialogue", "r2v",
         DIALOGUE_REF2V_PROMPT,
         dict(length=362, ref_image_count=2, ref_images=DIALOGUE_REF_IMAGES,
              out_prefix="Video/h3_r2v_dialogue",
              variant_note=_probe_note(
                  "does reference identity hold across seven turn changes",
                  "h3_text_to_video_dialogue.json",
                  "the same eight lines, with both speakers supplied as "
                  "stills instead of described.",
                  "identity on each cut -- the two references are far apart in "
                  "age, dress and palette, so a blend shows in one frame.",
                  "socket order is the label: <Picture 1> is the man and "
                  "<Picture 2> is the woman, so swapping them swaps who "
                  "speaks which lines.")),
         "the same exchange, both speakers from reference stills"),


        # --- PDD, the arms to actually render with ------------------------
        # The repo default: sage AND Sol.
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
              out_prefix="Video/text_to_video_pdd",
              variant_note=_probe_note(
                  "text to video at 8 steps via PDD",
                  "h3_text_to_video_pdd.json",
                  "the PDD LoRA at 8 evaluations, sage on, Sol absent.",
                  "identity and texture in the last third, where this "
                  "schedule takes its largest jump.",
                  "one converted file serves both step counts; the heads "
                  "are fused at load for whichever is asked.")),
         "text -> video + audio at 8 steps via PDD, sage on"),

        ("h3_text_to_video_pdd_4step.json", "texttovideopdd4step", "t2v", LONG_T2V_PROMPT,
         dict(pdd=True, sampler_name="euler",
              lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS_FAST,
              out_prefix="Video/text_to_video_pdd_4step",
              variant_note=_probe_note(
                  "text to video at 4 steps via PDD",
                  "h3_text_to_video_pdd.json",
                  "the PDD LoRA at 4 evaluations, sage on, Sol absent.",
                  "identity and texture in the last third, where this "
                  "schedule takes its largest jump.",
                  "one converted file serves both step counts; the heads "
                  "are fused at load for whichever is asked.")),
         "text -> video + audio at 4 steps via PDD, sage on"),

        ("h3_first_last_frame_to_video_pdd.json", "firstlastframetovideopdd", "i2v", None,
         dict(last_frame=True, **FL2V_CANVAS,
              pdd=True, sampler_name="euler",
              lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              out_prefix="Video/first_last_frame_to_video_pdd",
              variant_note=_probe_note(
                  "first+last frame to video at 8 steps via PDD",
                  "h3_first_last_frame_to_video_pdd.json",
                  "the PDD LoRA at 8 evaluations, sage on, Sol absent.",
                  "identity and texture in the last third, where this "
                  "schedule takes its largest jump.",
                  "one converted file serves both step counts; the heads "
                  "are fused at load for whichever is asked.")),
         "first+last frame -> video + audio at 8 steps via PDD, sage on"),

        ("h3_first_last_frame_to_video_pdd_4step.json", "firstlastframetovideopdd4step", "i2v", None,
         dict(last_frame=True, **FL2V_CANVAS,
              pdd=True, sampler_name="euler",
              lora=(PDD_FL2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS_FAST,
              out_prefix="Video/first_last_frame_to_video_pdd_4step",
              variant_note=_probe_note(
                  "first+last frame to video at 4 steps via PDD",
                  "h3_first_last_frame_to_video_pdd.json",
                  "the PDD LoRA at 4 evaluations, sage on, Sol absent.",
                  "identity and texture in the last third, where this "
                  "schedule takes its largest jump.",
                  "one converted file serves both step counts; the heads "
                  "are fused at load for whichever is asked.")),
         "first+last frame -> video + audio at 4 steps via PDD, sage on"),

        ("h3_image_ref_plus_text_to_video_pdd.json", "imagerefplustexttovideopdd", "r2v", _ref_prompt(images=True),
         dict(pdd=True, sampler_name="euler",
              lora=(PDD_REF2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              out_prefix="Video/image_ref_plus_text_to_video_pdd",
              variant_note=_probe_note(
                  "image references to video at 8 steps via PDD",
                  "h3_image_ref_plus_text_to_video_pdd.json",
                  "the PDD LoRA at 8 evaluations, sage on, Sol absent.",
                  "identity and texture in the last third, where this "
                  "schedule takes its largest jump.",
                  "one converted file serves both step counts; the heads "
                  "are fused at load for whichever is asked.")),
         "image references -> video + audio at 8 steps via PDD, sage on"),

        ("h3_image_ref_plus_text_to_video_pdd_4step.json", "imagerefplustexttovideopdd4step", "r2v", _ref_prompt(images=True),
         dict(pdd=True, sampler_name="euler",
              lora=(PDD_REF2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS_FAST,
              out_prefix="Video/image_ref_plus_text_to_video_pdd_4step",
              variant_note=_probe_note(
                  "image references to video at 4 steps via PDD",
                  "h3_image_ref_plus_text_to_video_pdd.json",
                  "the PDD LoRA at 4 evaluations, sage on, Sol absent.",
                  "identity and texture in the last third, where this "
                  "schedule takes its largest jump.",
                  "one converted file serves both step counts; the heads "
                  "are fused at load for whichever is asked.")),
         "image references -> video + audio at 4 steps via PDD, sage on"),


        ("h3_probe_ref2v_pdd.json", "r2v-pdd", "r2v", _ref_prompt(images=True),
         dict(pdd=True, dense_attn=True, sampler_name="euler", lora=(PDD_REF2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              length=243, out_prefix="Video/h3_probe_r2v_pdd",
              variant_note=_probe_note(
                  "does PDD hold ref2va identity at 8 steps",
                  "h3_probe_ref2v_pdd_headfree.json",
                  "the per-interval output heads are ON, which is the whole "
                  "PDD mechanism; the twin runs the same backbone and adaln "
                  "updates against the checkpoint's own heads.",
                  "identity on the reference subject, and texture in the last "
                  "third of the clip, where this schedule takes its biggest "
                  "jumps and where the fused heads differ most from the base.",
                  "if the two are indistinguishable, the heads are not what is "
                  "doing the work and the backbone LoRA alone is the cheaper "
                  "arm.")),
         "ref2va at 8 steps via Parallel Decoding Distillation"),

        # The control for the arm above. The measured gap between a fused head
        # and the checkpoint's own is 0.005 early and 0.015 at the last step
        # (docs/h3_pdd.md), so this is the arm that says whether that gap is
        # perceptible or merely real.
        ("h3_probe_ref2v_pdd_headfree.json", "r2v-pdd-headfree", "r2v",
         _ref_prompt(images=True),
         dict(pdd=True, dense_attn=True, sampler_name="euler", pdd_heads=False,
              lora=(PDD_REF2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              length=243, out_prefix="Video/h3_probe_r2v_pdd_headfree",
              variant_note=_probe_note(
                  "is the parallel-head machinery worth its complexity",
                  "h3_probe_ref2v_pdd.json",
                  "`patch_heads` is OFF, so the backbone and adaln updates "
                  "apply and the output heads stay the checkpoint's own.",
                  "the same places as its twin.",
                  "a visible loss here justifies the head machinery; no "
                  "visible loss says the backbone LoRA is the whole story.")),
         "PDD backbone only, the checkpoint's own output heads"),

        # Length sweep. The fused heads are indexed by time, not by call
        # count, so a longer clip changes the token budget and not the
        # schedule -- which is the property worth confirming rather than
        # assuming, because it is the one that would break silently.
        ("h3_probe_ref2v_pdd_345.json", "r2v-pdd-345", "r2v",
         _ref_prompt(images=True),
         dict(pdd=True, dense_attn=True, sampler_name="euler", lora=(PDD_REF2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              length=345, out_prefix="Video/h3_probe_r2v_pdd_345",
              variant_note=_probe_note(
                  "does PDD hold at the long end of the trained range",
                  "h3_probe_ref2v_pdd.json",
                  "345 frames instead of 243. Same schedule, more tokens.",
                  "drift late in the clip, and whether the boundary-residual "
                  "warning stays silent in the log.",
                  "the head selection is keyed on time, so length should not "
                  "move it at all; if it does, the keying is wrong.")),
         "PDD ref2va at the long end of the trained frame range"),

        ("h3_probe_ref2v_pdd_8s.json", "r2v-pdd-8s", "r2v",
         _ref_prompt(images=True),
         dict(pdd=True, dense_attn=True, sampler_name="euler", lora=(PDD_REF2VA_LORA, PDD_STRENGTH), steps=PDD_STEPS,
              length=192, out_prefix="Video/h3_probe_r2v_pdd_8s",
              variant_note=_probe_note(
                  "PDD at eight seconds",
                  "h3_probe_ref2v_pdd.json",
                  "192 frames, which is exactly 8.0 s on the 17k+5 grid at "
                  "24 fps, instead of 243.",
                  "the same places as its twin.",
                  "a length the grid hits exactly, so nothing is snapped and "
                  "the comparison is clean.")),
         "PDD ref2va at exactly eight seconds"),

        ("h3_probe_ref2v_turbo_pack.json", "r2v-turbo-pack", "r2v",
         _ref_prompt(images=True, video=True, video_audio=True,
                     video_role="swap", audio_role="copy"),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_image_count=1,
              turbo_pack=True,
              lora=(TURBO_PACK_LORA, TURBO_PACK_STRENGTH),
              steps=TURBO_PACK_STEPS, scheduler_name=TURBO_PACK_SCHEDULER,
              out_prefix="Video/h3_probe_r2v_turbo_pack",
              variant_note=_NOTE_TURBO_PACK),
         "character swap on ref2va with the adaln-touching turbo LoRA"),

        # The variant with the better prior. If ref2va's divergence really is
        # in the conditioning-modulation path, it binds hardest in the EARLY
        # steps, where composition and identity are still being decided. So
        # run those on the undistilled base and hand the tail to the distill:
        # the references get established by the model that understands them,
        # and the cheap steps go where the work is mostly refinement.
        #
        # `split_base_last=False` puts base FIRST. Its twin is the arm above,
        # which is the same LoRA with no split at all.
        ("h3_probe_ref2v_split_turbo_pack.json", "r2v-split-turbo-pack", "r2v",
         _ref_prompt(images=True, video=True, video_audio=True,
                     video_role="swap", audio_role="copy"),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_image_count=1,
              turbo_pack=True,
              lora=(TURBO_PACK_LORA, TURBO_PACK_STRENGTH),
              steps=TURBO_PACK_STEPS, scheduler_name=TURBO_PACK_SCHEDULER,
              split_at=SPLIT_AT, split_base_last=False,
              out_prefix="Video/h3_probe_r2v_split_turbo_pack",
              variant_note=_NOTE_TURBO_PACK_SPLIT),
         "base establishes the references, the distill finishes the clip"),

        ("h3_probe_reference_upscale.json", "r2v-noupscale", "r2v", _ref_prompt(images=True),
         dict(ref_upscale=False, out_prefix="Video/h3_probe_ref_noupscale",
              variant_note=_probe_note(
                  "does upscaling a small reference buy anything",
                  "h3_image_ref_plus_text_to_video.json",
                  "`allow_upscale` is OFF on both Append Picture nodes, "
                  "so references arrive at ComfyUI's own sizing instead of the "
                  "released pipeline's 2048 short edge.",
                  "Preflight's `references` line and percentage, then the "
                  "identity of the referenced subjects in the output. The "
                  "shipped graph spends roughly 13,900 more vision tokens on "
                  "the same two images.",
                  "Fewer tokens here, and a shorter sequence. Whether identity "
                  "is worse is the open question -- upscaling adds tokens, not "
                  "detail, and nobody has measured whether the checkpoint uses "
                  "them on an already-small source.")),
         "same references, without the reference pipeline's upscale"),

        ("h3_probe_square_canvas.json", "t2v-1to1", "t2v", LONG_T2V_PROMPT,
         dict(width=768, height=768,
              out_prefix="Video/h3_probe_square",
              variant_note=_probe_note(
                  "what an aspect ratio actually costs",
                  "h3_text_to_video.json",
                  "768x768 instead of 1344x768. Both are inside the trained "
                  "family; only the shape changed.",
                  "Preflight's sequence length on each, and render time. "
                  "Attention is O(S^2) and dominates the step.",
                  "About a third of the attention cost at the same frame "
                  "count, which is the largest single lever in this pipeline "
                  "-- larger than any kernel or sparsity setting.")),
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
        # Paired with `h3_probe_sol_on.json` deliberately: same canvas, same
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
         dict(sol_on=True, out_prefix="Video/h3_probe_sol_on_refs",
              variant_note=_probe_note(
                  "whether Sol-Attn's conditioning sink behaves at reference load",
                  "h3_probe_sol_on.json",
                  "reference images, against a t2v twin. Sol settings, canvas, "
                  "length and seed are identical; the sink grows from a few "
                  "hundred rows to thousands.",
                  "The `[sol_attn] conditioning sink` log line, with `verbose` "
                  "on. Read the START of the dense query range, not the size "
                  "of the change: a start of 0 means v2 did not engage, or the "
                  "audio span was never published and it fell back to v1 "
                  "silently. Then the video, for whether pinning references "
                  "exact actually preserves them.",
                  "KV blocks unchanged and the dense query range starting tens "
                  "of blocks in, where the t2v twin starts at 4. NOT predicted: "
                  "a speed win. References are exact rows Sol cannot sparsify, "
                  "so this arm should be SLOWER per token than the t2v twin "
                  "while still verifying the mechanism.")),
         "reference images with Sol-Attn ON -- the sink at reference load"),

        # The other Sol-Attn probe, and the older one. It exists so "is Sol
        # worth what it changes" stays answerable from a shipped artifact
        # rather than needing a hand-edit -- and that question is open in a way
        # the speed numbers do not settle, because nobody has weighed its
        # influence on the output against what it saves.
        # Read against h3_text_to_video.json, which is now sage-only.
        ("h3_probe_sol_on.json", "t2v-sol", "t2v", LONG_T2V_PROMPT,
         dict(sol_on=True, out_prefix="Video/h3_probe_sol_on",
              variant_note=_probe_note(
                  "whether Sol-Attn earns its influence on the output",
                  "h3_text_to_video.json",
                  "Sol-Attn enabled, at SOL_RECOMMENDED_CUDA. Its twin is sage-only, "
                  "which is what every shipped graph is now.",
                  "Wall clock AND the video. Sol changes what the model "
                  "computes -- it is sparse attention, not a faster exact "
                  "kernel -- so a speed win that costs output quality is not a "
                  "win. Watch motion and drift, the axes fp16-PV was chosen "
                  "on, since those are where an approximation shows first.",
                  "Faster, by an amount that grows with sequence length. What "
                  "is NOT predicted is the output being indistinguishable: "
                  "the sparse kernel skips blocks the exact one attends, and "
                  "whether that is visible at H3's shapes is exactly what has "
                  "never been judged here.")),
         "Sol-Attn on, against the sage-only twin"),

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
              out_prefix="Video/h3_probe_sol_on_all_refs",
              variant_note=_probe_note(
                  "what Sol-Attn does when every reference type is present",
                  "h3_ref_image_video_audio.json",
                  "Sol-Attn enabled, at SOL_RECOMMENDED_CUDA. Its twin is the "
                  "same references sage-only.",
                  "The `[sol_attn] conditioning sink` line with `verbose` on, "
                  "and then the video. Reference rows are exact keys at any "
                  "tau, so what to watch is whether the SUBJECTS survive -- "
                  "face and identity against the reference images, motion "
                  "against the reference video, and the soundtrack.",
                  "A large sink and a small dense-query span. NOT a speed win "
                  "proportional to the token count: exact reference rows are "
                  "work Sol cannot skip, so this arm should be slower per "
                  "token than a text-only one while still being the case worth "
                  "getting right.")),
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
              out_prefix="Video/h3_probe_cache_easy",
              variant_note=_probe_note(
                  "whether TeaCache-family step reuse pays on H3 at 16 steps",
                  "h3_probe_sol_on_all_refs.json",
                  "identical except the EasyCache node between the attention "
                  "chain and the sampler, at CACHE_NODE defaults.",
                  "the EasyCache verbose lines in the server log -- how many "
                  "of the 16 steps were reused. A run without that count is "
                  "uninterpretable. Then the video, against the twin's.",
                  "NVLabs' 4090 H3 runtime attributes 3.18x of its speedup "
                  "to caching at 50 steps; at 16 steps with the first ~15% "
                  "and last ~5% forced dense, at most 12 forwards are "
                  "skippable, so expect far less. On er_sde the per-step "
                  "re-noising inflates input deltas, so a zero-reuse result "
                  "here is a sampler artifact until re-run on a "
                  "deterministic sampler.")),
         "the all-refs Sol arm plus EasyCache step reuse"),

        # The euler pair, owner-requested 2026-08-18. Same workload as the
        # two graphs above with only the sampler changed: euler is
        # deterministic, so it is the arm where step caching works at the
        # stock threshold (measured the same day on res_multistep: 7 of 16
        # steps reused, 1.74x on the sampler, where the shipped er_sde
        # reused nothing at 0.2 -- bench/results/2026-08-18_cache_arms.jsonl)
        # and where a cache-on/off pair is a valid numeric A/B under the
        # CLAUDE.md deterministic-sampler rule. Two graphs, not one, so the
        # pair varies exactly the cache node.
        ("h3_probe_euler.json", "r2v-all-euler", "r2v",
         _ref_prompt(images=True, video=True, video_audio=True, audio=True),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_audio=True, sol_on=True,
              sampler_name="euler",
              out_prefix="Video/h3_probe_euler",
              variant_note=_probe_note(
                  "what the all-refs workload looks like on a deterministic "
                  "sampler from the euler family",
                  "h3_probe_sol_on_all_refs.json",
                  "identical except KSamplerSelect: euler instead of er_sde.",
                  "the clip, against the er_sde twin's -- sampler swaps "
                  "cannot be pixel-compared, so the question is whether the "
                  "brief survives, not whether frames match.",
                  "same per-step cost (sampler choice measured speed-neutral "
                  "2026-08-18); any difference is look, not wall time.")),
         "the all-refs workload on euler -- the deterministic-sampler arm"),

        ("h3_probe_euler_cache.json", "r2v-all-euler-cache", "r2v",
         _ref_prompt(images=True, video=True, video_audio=True, audio=True),
         dict(**REF_VIDEO_BUDGET, ref_video=True, ref_audio=True, sol_on=True,
              sampler_name="euler", cache=CACHE_NODE,
              out_prefix="Video/h3_probe_euler_cache",
              variant_note=_probe_note(
                  "whether euler gets the deterministic-sampler cache payoff",
                  "h3_probe_euler.json",
                  "identical except the EasyCache node at CACHE_NODE "
                  "defaults.",
                  "the EasyCache verbose skip count in the server log, then "
                  "the clip against the twin's -- this pair IS seed-pairable "
                  "(same sampler, deterministic).",
                  "if euler behaves like res_multistep, roughly 7 of 16 "
                  "steps reused; a lower count is a finding about euler's "
                  "trajectory, not a harness failure.")),
         "the euler arm plus EasyCache -- the cache-payoff twin"),

        # Sol-Attn ON with an input image rather than references. Keyframe
        # `cond` rows land in the sink too, so this is the third sink shape:
        # text-only, reference-heavy, and keyframe.
        ("h3_probe_sol_on_i2v.json", "i2v-sol", "i2v", None,
         dict(sol_on=True, out_prefix="Video/h3_probe_sol_on_i2v",
              variant_note=_probe_note(
                  "whether Sol-Attn preserves a supplied first frame",
                  "h3_first_frame_to_video.json",
                  "Sol-Attn enabled, at SOL_RECOMMENDED_CUDA. Its twin is the "
                  "same first frame sage-only.",
                  "Whether the opening frame still matches the image you "
                  "supplied, and whether the clip drifts away from it faster "
                  "than the sage-only twin does. The keyframe rows sit in the "
                  "sink, so they are exact keys -- drift here would be the "
                  "video losing them, not the conditioning being dropped.",
                  "Close to the twin at the opening and diverging later, since "
                  "that is where a block-sparse router has had the most steps "
                  "to accumulate. Unmeasured: nobody has run Sol on a keyframe "
                  "graph at all.")),
         "first frame + text, with Sol-Attn ON"),

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
        # `archive/single_frame.py`. `_image_graphs()` and everything it reaches
        # is left intact rather than deleted so restoring the lane is
        # un-parking this one line -- but nothing downstream may assume the
        # directory exists while it is parked. See `docs/h3_image_editing.md`.
        # *_image_graphs(),

        ("h3_probe_head_chunks.json", "t2v-chunk4", "t2v", LONG_T2V_PROMPT,
         dict(head_chunks=4, out_prefix="Video/h3_probe_chunk4",
              variant_note=_probe_note(
                  "trading launches for VRAM headroom",
                  "h3_text_to_video.json",
                  "`head_chunks` 4 on the SageAttention node instead of 1.",
                  "Peak VRAM, and wall clock. Nothing about the output should "
                  "change: chunking splits the heads, it does not alter the "
                  "arithmetic.",
                  "Peak attention drops from 2862 MiB to 2645 at the default "
                  "canvas, because chunking rules out the v clone that only "
                  "pays unchunked. It costs 4 kernel launches per call, "
                  "measured at a ~2.6% wall-clock ceiling on a 24 GB 4090. "
                  "Take it to fit a render that otherwise will not fit, not "
                  "for speed.")),
         "the same render with the heads in 4 groups"),
    )

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
    for fname, label, task, prompt, extra, note in GRAPHS:
        is_image = bool(extra.get("single_frame", False))
        # An SLA-router arm replaces the sage+Sol chain outright: no sage
        # node, no Sol node, and the chain assert warn-only. One graph kind,
        # exempted by mechanism in check_attention_defaults.
        router = extra.get("sla_router") is not None
        # An arm that patches the DiT's self-attention not at all: no sage,
        # no Sol, whatever
        # kernel ComfyUI resolves on its own. Distinct from the repo's usual
        # "dense" (sage alone) and from the router arm. It exists for probes
        # whose subject is a numerical mechanism somewhere else in the model:
        # both sage and Sol change attention numerics, so leaving them in puts
        # two approximations in the path of an experiment about a third.
        # "none" wires neither sage nor Sol; "sage" wires sage with Sol ABSENT.
        # Absent rather than bypassed, for PDD: Sol skips attention adaptively
        # per step, which is incoherent against a fixed fused block schedule,
        # and a bypassed node in the graph is an invitation to switch it on.
        dense = extra.get("dense_attn", False)
        dense_mode = ("none" if dense is True else dense) or None
        sol_on = False if (is_image or router or dense_mode) else bool(extra.get("sol_on", True))
        rest = {k: v for k, v in extra.items()
                if k not in ("sol_on", "dense_attn")}
        wf = build_ui(task, sage=(dense_mode == "sage") if dense_mode else not router,
                      preview=True,
                      sol=(sol_for_steps(SOL_RECOMMENDED_CUDA,
                                         extra.get("steps", SAMPLING["steps"]))
                           if not (is_image or router or dense_mode) else None),
                      sol_enabled=sol_on, prompt=prompt,
                      title=f"h3-{label}-" + ("sla-router" if router else
                                              "dense" if dense_mode == "none" else
                                              "sage" + ("-sol" if sol_on else "")),
                      **{"length": LONG_LENGTH, **rest})
        p = _graph_dir(out, extra) / fname
        written.append((label, "ui", p, wf))
        print(f"  {p.name}: {note}")

    # API-format copies of the same graphs, for driving a render over /prompt
    # without a browser. Same builder inputs, so they cannot describe a
    # different configuration than the set above.
    for fname, label, task, prompt, extra, _note in GRAPHS:
        is_image = bool(extra.get("single_frame", False))
        router = extra.get("sla_router") is not None
        dense = extra.get("dense_attn", False)
        dense_mode = ("none" if dense is True else dense) or None
        sol_on = False if (is_image or router or dense_mode) else bool(extra.get("sol_on", True))
        api_extra = {k: v for k, v in extra.items()
                     if k not in ("variant_note", "sol_on", "dense_attn")}
        wf = build_api(task, sage=(dense_mode == "sage") if dense_mode else not router,
                       prompt=prompt,
                       sol=(sol_for_steps(SOL_RECOMMENDED_CUDA,
                                          extra.get("steps", SAMPLING["steps"]))
                            if sol_on else None),
                       **{"length": LONG_LENGTH, **api_extra})
        p = _graph_dir(out, extra) / fname.replace(".json", "_api.json")
        written.append((label, "api", p, wf))

    # Bench copies carrying MiniMaxH3ProvenanceStamp. Deliberately NOT the
    # shipped graphs: the stamp reads another pack's closure internals, so it
    # breaks when that pack changes, and a bench is where breakage is cheap.
    # API-only, so cross_check skips them (it needs both formats to compare).
    bench = out / "bench"
    bench.mkdir(parents=True, exist_ok=True)
    for fname, task, prompt in (
        ("h3_text_to_video_stamped_api.json", "t2v", LONG_T2V_PROMPT),
        ("h3_image_ref_plus_text_to_video_stamped_api.json", "r2v", None),
        ("h3_first_frame_to_video_stamped_api.json", "i2v", None),
    ):
        wf = build_api(task, sage=True, length=LONG_LENGTH,
                       sol=None, prompt=prompt, stamp=True)
        p = bench / fname
        written.append((f"{task}-stamped", "api", p, wf))

    def flush():
        """Write every graph. Called only once nothing has objected.

        Writes used to happen inline, as each graph was built, and validation
        ran afterwards over what was already on disk. So a red build still
        SHIPPED its graphs and merely reported a nonzero exit -- which is how
        `MiniMaxH3AppendRefImage` corrupted 40 UI graphs on 2026-08-25 and
        stayed corrupted: the failure was printed every time anyone
        regenerated, and the bad files were already written by then.

        Nothing is written now until the cross-check, the validators and the
        staleness verdict have all passed. A failed build leaves the tree
        exactly as it found it.
        """
        for _t, _f, path, doc in written:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
            print(f"wrote {path.name}")

    # Cross-check the two formats of each task describe the same graph. The
    # per-format validators below only prove each is well-formed against
    # object_info; nothing there would notice the UI graph carrying a
    # the Sol node the API graph lacks, which is exactly the state this file
    # was in before 2026-08-06.
    drift = cross_check(written)
    if drift:
        print("\nUI/API DRIFT:")
        for x in drift:
            print("  " + x)
        return 1
    print("UI/API cross-check: same node counts and settings")

    if args.no_validate:
        flush()
        return 0
    oi = load_object_info(args.object_info)
    errs = []
    for task, fmt, p, wf in written:
        errs += (validate_api if fmt == "api" else validate_ui)(wf, oi, p.name)
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
