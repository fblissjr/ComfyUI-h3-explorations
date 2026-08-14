#!/usr/bin/env python3
"""End-to-end A/B: a real MiniMax H3 render with the sage node in and out.

The per-module bench (`bench_minimax_attn.py`) measures one Attention in
isolation. This one submits an actual render to a running ComfyUI and
reports what the user experiences, which is the only number that settles
whether the kernel win survives contact with a 21 GB checkpoint on a
24 GB card.

Three things are measured per run, all from ComfyUI's websocket
node-transition events except the last:

  - **sampler time** -- `SamplerCustomAdvanced` alone, isolated from text
    encoding and VAE decode. This is where an attention speedup has to
    show up.
  - **decode time** -- `VAEDecode` alone. Attention arms do not move it,
    which is exactly why it is worth printing: it is the denominator that
    decides how much of a render any attention work can reach. It becomes
    the measured quantity when `--video-vae` is given more than one VAE.
  - **total wall-time**, from submit to history. What actually changes for
    the user, and always a smaller ratio than whichever stage an arm
    moves, because the other stages are unaffected.

Method notes that matter for trusting the result:

  - The first run is a warmup and is discarded. It pays model load,
    Triton autotune for every new shape, and the Qwen3-VL-32B text
    encode. Including it would swamp everything else.
  - Arms alternate (A B A B ...) rather than running in blocks, so any
    drift in clocks, thermals or allocator state is shared rather than
    attributed to whichever arm ran second.
  - The graph is built here rather than converted from the bundled UI
    templates: two of those hide the sampler stack inside a subgraph,
    and hand-converting a subgraph to API format is a good way to
    measure something subtly different from what the template runs.
    Settings are copied from `video_minimax_h3_i2v.json`.

    ./bench/bench_e2e_h3.py --runs 3
    ./bench/bench_e2e_h3.py --runs 3 --width 768 --height 768 --steps 10

  VAE A/B -- one arm, two VAEs, crossed so they alternate. Note a VAE swap
  invalidates the sampler too (MiniMaxH3ImageToVideo takes the same VAE for
  keyframe encoding), so both sides pay a full sample; keep steps low, the
  decode column is the one being read:

    ./bench/bench_e2e_h3.py --arms sage+sol --runs 2 --length 124 --steps 6 \
        --video-vae minimax_h3_video_vae_fp16.safetensors,minimax_h3_video_vae_int8_convrot.safetensors

Needs a running ComfyUI with the MiniMax H3 models installed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path

# Settings lifted from the bundled i2v template so this measures the
# configuration people actually run.
DEFAULTS = dict(
    unet="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    clip="qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    video_vae="minimax_h3_video_vae_fp16.safetensors",
    audio_vae="minimax_h3_audio_vae_fp32.safetensors",
    sampler="res_multistep",
    scheduler="simple",
    steps=20,
    width=1344,
    height=768,
    length=73,
    fps=24.0,
)

# Long-clip prompt, used when the request is long enough to need one.
#
# Two reasons this exists rather than stretching PROMPT. First, format: past
# a few seconds MiniMax's guide wants the timeline carried by numbered shots
# with explicit cut times and the three core fields, not one run-on
# description -- a 15-second request against a 6-second prompt leaves the
# model to invent twelve seconds of nothing, which is its own confound.
#
# Second, and the reason the content is what it is: docs/SOLATTN.md flags that
# the earlier quality comparison ran on slow camera moves and diffuse fog,
# which is close to the worst case for *noticing* a block-sparse artifact.
# A router that drops the wrong block shows up in fast motion, in fine
# repeated detail, and in audio transients. So this deliberately carries
# all three -- a whip pan, a brick facade and railings, rain texture, and
# sharp percussive sound -- to give any degradation somewhere to be seen.
# A prompt that hides artifacts makes the sparse arm look free when it
# is not.
PROMPT_LONG = (
    "integrated_multimodal_description: [Shot 1] Live-action, cinematic, "
    "handheld, shallow depth of field. A medium shot frames a courier in a "
    "soaked red jacket standing over a bicycle at a city crosswalk in heavy "
    "evening rain, wet asphalt throwing back the signal lights, a brick "
    "facade with iron railings filling the background. The camera tracks "
    "right at medium amplitude and moderate speed as she snaps her helmet "
    "strap and pushes off.\n"
    "[Shot 2] At 00:04.000, the shot cuts to a low tracking shot running "
    "alongside the spinning front wheel, spokes flickering, spray coming off "
    "the tyre, painted lane markings streaming past underneath.\n"
    "[Shot 3] At 00:08.000, the camera whip pans up to a wide shot of the "
    "street as she cuts between two parked cars, pigeons scattering off the "
    "railings, neon shopfront signs reflected in the puddles.\n"
    "[Shot 4] At 00:11.500, the shot changes to a close shot of her face "
    "under the helmet, rain streaking across the lens, as she glances back "
    "over her shoulder and then forward again, breathing hard.\n\n"
    "overall_soundscape: steady heavy rain on asphalt and metal, tyre hiss "
    "through standing water, the click and rattle of a bicycle chain, spokes "
    "ticking, a car horn twice in the middle distance, wings clattering as "
    "the pigeons take off, her breathing close and rhythmic under the "
    "helmet.\n\n"
    "non_diegetic_music: none."
)

# A prompt shaped the way MiniMax's own writing guide recommends:
# style and composition first, then subject, scene, camera motion,
# action, then soundscape.
PROMPT = (
    "Live-action, cinematic, shallow depth of field. A medium-wide shot "
    "frames a lone lighthouse keeper on a wet stone balcony at dawn, "
    "wearing a heavy oilskin coat, the lamp housing glowing behind them. "
    "Grey-blue sea fog rolls past below, gulls crossing the frame. "
    "The camera pushes in slowly with small amplitude as the keeper "
    "raises a brass telescope, holds it steady, then lowers it and turns "
    "toward the light.\n\n"
    "Audio: low sea swell and wind against stone, a distant foghorn twice, "
    "gulls calling overhead.\n\n"
    "No dialogue, no text overlays, no cuts."
)


def _split_arms(spec):
    """Split --arms on commas that separate arms, not ones inside brackets.

    A plain `spec.split(",")` shreds `sage+sol[int8_qk=0,int8_pv=0]` into
    two fragments, neither of which parses. Depth-tracking keeps the
    overrides together.
    """
    out, depth, cur = [], 0, []
    for ch in spec:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return out


def _is_adhoc(name):
    return name.endswith("]") and "[" in name


# Which Sol-Attn node the sol arms build. Set once from --sol-backend, before
# any arm is resolved. A module global rather than a parameter threaded through
# resolve_arm/build_prompt because it is a property of the RUN, not of an arm:
# mixing backends inside one A/B would compare two kernels and call it a knob.
#
# Default flipped triton -> cuda on 2026-08-14. The two backends were shown
# arithmetically equivalent against the algorithm's own eager reference (cuda
# 0.999919, triton int8 0.999885, each in its own centroid_tail mode), so there
# is no accuracy argument for Triton, and upstream reports cuda at 1.4x
# end-to-end at the same tau. Triton stays reachable, and stays installed,
# because every number recorded before this date was taken on it and
# check_solattn_correctness.py grades the two against the same oracle -- a
# cross-check that only exists while both are present.
SOL_BACKEND = "cuda"


def sol_node():
    """(class_type, knob defaults) for the active backend.

    Resolved lazily: the knob dicts are imported from h3_config further down
    this file, so a module-level table here would NameError at import.
    """
    from h3_config import SOL_BASELINE_124F, SOL_CUDA_DEFAULTS
    return {
        "triton": ("SolAttnPatch", SOL_BASELINE_124F),
        "cuda": ("SolAttnMiniMax", SOL_CUDA_DEFAULTS),
    }[SOL_BACKEND]


def sol_knobs():
    """The active backend's knob set, for typing and validating overrides."""
    return sol_node()[1]


def resolve_arm(name):
    """(sage on?, SolAttn overrides) for a named arm or an ad-hoc spec.

    Named arms in ARMS stay the vocabulary for anything worth repeating.
    Ad-hoc specs exist because sweeping a continuous knob -- tau, the
    sigma window -- would otherwise mean editing ARMS once per point and
    re-committing, which is how a sweep ends up undocumented. Form:

        sage+sol[tau=1.6]                 sage on, one override
        sage+sol[tau=2.0,int8_qk=1]       several
        sol[start_percent=0.1]            SolAttn without sage
        kj+sol[tau=1.6]                   KJNodes' H3 patch instead of ours

    A `kj` prefix swaps the patching surface for KJNodes'
    MiniMaxH3MemoryEfficientSageAttentionPatch. Both patch the same 50
    attention forwards and both call this install's sage kernels; they
    differ in that ours routes through `sageattn_consume` (releasing q/k/v
    as their quantized forms appear) and leaves `smooth_k` off, where
    KJNodes hand-rolls the per-arch quant sequence and turns `smooth_k` on.

    Values are typed by SOL_DEFAULTS, so `int8_qk=1` becomes True and
    `tau=1.6` a float. A key not in SOL_DEFAULTS is an error rather than
    a silently-ignored typo -- a misspelled knob would otherwise read as
    "this lever does nothing".
    """
    if not _is_adhoc(name):
        return ARMS[name]
    base, spec = name[:-1].split("[", 1)
    overrides = {}
    for pair in spec.split(","):
        if not pair.strip():
            continue
        k, _, v = pair.partition("=")
        k, v = k.strip(), v.strip()
        if k not in sol_knobs():
            other = SOL_CUDA_DEFAULTS if SOL_BACKEND == "triton" else SOL_DEFAULTS
            hint = (f"; {k!r} is a {'CUDA' if SOL_BACKEND == 'triton' else 'Triton'}-only "
                    f"knob and this run is --sol-backend {SOL_BACKEND}"
                    if k in other else "")
            raise SystemExit(f"unknown SolAttn knob {k!r}{hint}; "
                             f"known: {sorted(sol_knobs())}")
        proto = sol_knobs()[k]
        if isinstance(proto, bool):
            overrides[k] = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(proto, int):
            overrides[k] = int(v)
        elif isinstance(proto, float):
            overrides[k] = float(v)
        else:
            overrides[k] = v
    # An ad-hoc base that names a known arm INHERITS it, so
    # `shipped[centroid_tail=0]` means "the shipped config with that one knob
    # moved" rather than "the node defaults with that one knob moved". Without
    # this a sweep silently measures a different config than the one it is
    # named after -- and `shipped[...]` would also have come out sage-OFF,
    # because "shipped" does not start with "sage".
    if base in ARMS:
        base_sage, base_sol = ARMS[base]
        if base_sol is None:
            raise SystemExit(f"arm {base!r} has no Sol settings to override")
        return base_sage, dict(base_sol, **overrides)
    return ("kj" if base.startswith("kj") else base.startswith("sage")), overrides


def pick_prompt(cfg):
    """PROMPT_LONG once the request is long enough to need a shot timeline.

    The threshold is where PROMPT's single shot stops covering the runtime:
    it describes one continuous ~6 s beat, so anything past roughly twice
    that is asking the model to fill time the prompt never mentions. Cut
    times in PROMPT_LONG run to 00:11.500, so it needs at least that much
    clip to make sense.
    """
    seconds = cfg["length"] / cfg["fps"]
    return PROMPT_LONG if seconds >= 12.0 else PROMPT


def build_prompt(cfg, *, sage, seed, sol=None, head_chunks=1, ffn_chunks=1):
    """API-format graph.

    `sage` inserts our node between UNETLoader and the two MODEL consumers.
    `sol`, when given a dict of SolAttnPatch settings, chains SolAttn after
    it -- that order matters: SolAttn walks the model's existing object
    patches and composes with the attention forwards it finds, so it has to
    run second to see ours. Reversed, ours would overwrite its patch.
    """
    g = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": cfg["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": cfg["clip"], "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": cfg["video_vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": cfg["audio_vae"]}},
        "5": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0],
                         "prompt": pick_prompt(cfg),
                         "width": cfg["width"], "height": cfg["height"],
                         "length": cfg["length"]}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": cfg["sampler"]}},
        "8": {"class_type": "BasicScheduler",
              "inputs": {"model": None, "scheduler": cfg["scheduler"],
                         "steps": cfg["steps"], "denoise": 1.0}},
        "9": {"class_type": "BasicGuider",
              "inputs": {"model": None, "conditioning": ["5", 0]}},
        "10": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0],
                          "sigmas": ["8", 0], "latent_image": ["5", 1]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        "13": {"class_type": "CreateVideo",
               "inputs": {"images": ["11", 0], "fps": cfg["fps"], "audio": ["12", 0]}},
        "14": {"class_type": "SaveVideo",
               "inputs": {"video": ["13", 0], "filename_prefix": "video/h3_sage_ab",
                          "format": "auto", "codec": "auto"}},
    }
    model_src = ["1", 0]
    if sage == "kj":
        # KJNodes' patch as the attention surface instead of ours. It takes no
        # options -- no mode, no token-refiner switch -- and it calls this
        # install's sage kernels either way, so this arm swaps the wrapper,
        # not the kernel. The two differ in that we route through
        # sageattn_consume and leave smooth_k off where KJNodes hand-rolls
        # the quant sequence and enables it.
        g["20"] = {"class_type": "MiniMaxH3MemoryEfficientSageAttentionPatch",
                   "inputs": {"model": model_src}}
        model_src = ["20", 0]
    elif sage:
        # From SAGE_NODE, not a literal. This was `mode="auto"` until
        # 2026-08-14, and it had been wrong since the fp16 flip landed in
        # h3_config.py and the graphs on 2026-08-13: `auto` resolves to
        # fp8_cuda++, the FASTEST kernel, where the shipped config is
        # "fp16 (most accurate)" and costs ~1.58x for 2.7x the accuracy.
        #
        # So every e2e arm measured between those dates compared against a
        # sage baseline nobody ships, and the direction of the error is not
        # neutral: fp8 sage is fast, so a Sol arm measured against it looks
        # LESS impressive than against the real one. Reading a shared
        # constant rather than repeating it is the fix, which is the same
        # lesson check_generator_constants.py exists to enforce for the
        # generator -- and the bench was never covered by it.
        g["20"] = {"class_type": "MiniMaxH3SageAttention",
                   "inputs": {"model": model_src,
                              **dict(SAGE_NODE, head_chunks=head_chunks)}}
        model_src = ["20", 0]
    if ffn_chunks > 1:
        # KJNodes'. Deliberately not reimplemented: it patches mlp.forward,
        # which nothing of ours touches, so unlike head chunking there is no
        # conflict to resolve by owning it. Placed before SolAttnPatch only
        # for consistency with the chain order; it patches a different module
        # and composes with any of these in any order.
        g["22"] = {"class_type": "MiniMaxChunkFeedForward",
                   "inputs": {"model": model_src, "chunks": ffn_chunks,
                              "seq_threshold": 4096}}
        model_src = ["22", 0]
    if sol is not None:
        # Node id 21 for both backends: they are alternatives, never both in
        # one graph, and keeping the id stable keeps the timing breakdown
        # comparable across a --sol-backend switch.
        class_type, defaults = sol_node()
        g["21"] = {"class_type": class_type,
                   "inputs": {"model": model_src, **defaults, **sol}}
        model_src = ["21", 0]
    g["8"]["inputs"]["model"] = model_src
    g["9"]["inputs"]["model"] = model_src
    return g


# SolAttnPatch's settings, pinned so an arm only has to name what it changes
# and the rest cannot drift with the node.
#
# Pinning is load-bearing and nearly failed here: SolAttn changed three of
# these underneath us. `int8_qk` and `int8_pv` now default on upstream and
# `morton_curve` now defaults to "2d_frame". Any knob missing from this dict
# silently takes the node's current default, so an arm named "sol" would
# quietly have meant different things before and after that release, and the
# ratios would not have been comparable across sessions.
#
# These come from workflows/h3_config.py, which the graph builder imports
# too. They used to be a second copy living here, and the two went out of
# sync the moment either was edited -- so a bench arm named "sol" and the
# workflow you would actually open were different configurations, and the
# number described something nobody ran.
#
# SOL_DEFAULTS stays the 124-frame baseline so recorded measurements remain
# comparable. SOL_RECOMMENDED is what the shipped workflows run; arms opt
# into it by name rather than by editing the baseline.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflows"))
from h3_config import (  # noqa: E402
    SAGE_NODE,
    SOL_BASELINE_124F as SOL_DEFAULTS,
    SOL_CUDA_DEFAULTS,
    SOL_RECOMMENDED,
    SOL_RECOMMENDED_CUDA,
)

# Below this, a Sol-Attn arm cannot show anything.
#
# In TOKENS, not frames. Frames were the first spelling here and it was wrong:
# video tokens are `latent_t * (w//32) * (h//32)`, so the canvas is half the
# quantity. 250 frames at 1344x768 is 72,576 video tokens; the same 250 frames
# at 832x768 is 44,928 -- a third less, and on the wrong side of the floor. A
# frame-count guard passes that second run and lets it read as a null result.
#
# 60k is the lowest figure upstream names as working ("surprisingly well on
# only 60k tokens"). His separate statement that nothing is visible under
# ~250-300 frames implies 72k-88k at the 1344x768 he works at, so this is the
# permissive end of his range on purpose -- the guard should fire only when a
# run is definitely uninformative, not merely small.
#
# Video tokens only. The real attention sequence is the packed
# [text][cond][ref][audio][video], so this under-counts, which is the safe
# direction for a floor.
SOL_TOKEN_FLOOR = 60_000


def video_tokens(width, height, length):
    """Video tokens for a canvas and pixel-frame count.

    `latent_t = ((n - 5) // 17) * 5 + 2` is the inverse of preflight.py:155,
    and reproduces both figures this repo already records: 362 frames -> 107
    latent frames, 124 -> 37.
    """
    n = int(length)
    latent_t = ((n - 5) // 17) * 5 + 2 if n > 5 else 2
    return latent_t * (int(width) // 32) * (int(height) // 32)

# name -> (sage on?, SolAttn overrides or None)
ARMS = {
    "off":       (False, None),
    "sage":      (True, None),
    # KJNodes' H3 patch as the attention surface. Same 50 forwards, same
    # installed kernels; the wrapper differs.
    "kj":        ("kj", None),
    "kj+sol":    ("kj", {}),
    "kj+sol+int8": ("kj", {"int8_qk": True, "int8_pv": True}),
    "sol":       (False, {}),
    "sage+sol":  (True, {}),
    # Exactly what workflows/h3_text_to_video.json runs. This is the arm to
    # quote a render time from, because it is the only one whose settings
    # are the ones you would actually open. Everything else here is a probe
    # that isolates one knob against the 124-frame baseline.
    # "shipped" means what workflows/ actually wires, which since 2026-08-14
    # is the CUDA node at SOL_RECOMMENDED_CUDA. The Triton config it replaced
    # is still reachable as `shipped_triton`, which needs --sol-backend triton
    # and exists to reproduce a pre-migration number rather than to be run.
    "shipped":   (True, dict(SOL_RECOMMENDED_CUDA)),
    "shipped_triton": (True, dict(SOL_RECOMMENDED)),
    "sage+sol+morton": (True, {"morton": True}),
    # int8_qk puts SolAttn's exact branch on INT8 QK instead of fp16, which
    # its own tooltip says helps at tau<=1.5 -- we run tau=1.2. Without it
    # the stacked arms are not purely additive: sage quantizes, SolAttn's
    # kept blocks do not.
    "sage+sol+int8qk": (True, {"int8_qk": True}),
    # verbose logs each sparse/dense routing decision. Not a timing arm --
    # it exists to prove SolAttn engaged at all, since a failed compose
    # degrades to dense silently and reads as "sparsity did not help".
    "sage+sol+verbose": (True, {"verbose": True}),
    # Morton is close to free and measured best of the defaults, so the
    # tuned arms build on it rather than on bare sol.
    "sage+sol+morton+int8qk": (True, {"morton": True, "int8_qk": True}),
    # int8_pv is the other half of the int8 win -- SolAttn's exact branch
    # runs P@V in INT8 as well as QK, and upstream's note is that the two
    # cost the same. It only applies when int8_qk is on, so the pair moves
    # together. This is what upstream now ships on by default and what the
    # 124-frame evaluation predates.
    "sage+sol+int8": (True, {"int8_qk": True, "int8_pv": True}),
    # 2d_frame Z-orders within each frame and leaves frame order alone.
    # It became upstream's default because H3's frame spacing is not
    # uniform, which is a length-dependent failure -- so this arm matters
    # more at 362 frames (latent_t 107) than at the 124 frames (latent_t 37)
    # the earlier evaluation ran, where morton measured neutral-to-slightly-good.
    "sage+sol+morton2d": (True, {"morton": True, "morton_curve": "2d_frame"}),
    # int8 + the frame-local morton curve, on our pinned baseline. NOT a
    # reproduction of upstream's shipped defaults -- it inherits tau=1.2 and
    # sink_conditioning="exact_kv" from SOL_DEFAULTS, where upstream ships
    # 1.3 and "exact_kv_and_rows". Named "current" originally, which claimed
    # more than it tested; it is an isolation arm for morton-on-top-of-int8.
    "sage+sol+int8+morton2d": (True, {"int8_qk": True, "int8_pv": True,
                                      "morton": True, "morton_curve": "2d_frame"}),
    # What a user actually gets from dropping in the current node untouched.
    # Every knob at upstream's own default, overriding our pinned baseline
    # where the two differ (tau, sink_conditioning).
    "sage+sol+upstream_defaults": (True, {"tau": 1.3, "int8_qk": True,
                                          "int8_pv": True, "morton": True,
                                          "morton_curve": "2d_frame",
                                          "sink_conditioning": "exact_kv_and_rows"}),
    # H3's audio is ~250-400 rows inside a ~38k packed sequence -- thin
    # enough for a block-sparse router to drop. exact_kv_and_rows runs
    # those query rows dense so the generated audio stream stays exact,
    # at ~20% cost by its own tooltip. The knob behind "it helps audio".
    "sage+sol+morton+audio": (True, {"morton": True,
                                     "sink_conditioning": "exact_kv_and_rows"}),

    # --- start_percent sweep, added 2026-08-14 ------------------------------
    #
    # `start_percent` is the only knob in SOL_RECOMMENDED that was never
    # measured. 0.2 is the paper's number and it was carried straight through;
    # every other knob there has a paragraph of evidence above it in
    # h3_config.py. Upstream's report is that a later start affects motion
    # least, which would make this the first lever to reach for when quality
    # needs clawing back -- but that is the author's observation, not a
    # measurement of ours, and it is the reason these arms exist.
    #
    # They extend `shipped` (SOL_RECOMMENDED), NOT SOL_DEFAULTS. Sweeping this
    # against the 124-frame baseline would measure start_percent at tau=1.2
    # and sink_conditioning="exact_kv", which is not what anyone runs -- and
    # h3_config.py already carries the receipt for that mistake: "a knob
    # validated at one setting of another knob is not validated". If tau ever
    # moves, these arms move with it, which is the point of deriving them.
    #
    # RUN THESE LONG. `--length` defaults to 73 and the whole frontier table
    # in docs/SOLATTN.md was measured at 124; upstream is explicit that the
    # gains only appear at high token counts -- large canvas, long duration,
    # or many references -- and that the relative gain grows with size, with
    # its own 1.4x figure taken around 500 frames. A start_percent sweep at
    # the default length measures the regime where Sol-Attn has nothing to
    # find, and would read as "this knob does nothing".
    #
    # What each is predicted to show, written down before running so the
    # result can contradict it: time should fall roughly linearly as
    # start_percent drops (more steps inside the sparse window), and the
    # moving-content artifact -- a small persistent object dissolving partway
    # through a clip, documented under Quality in docs/SOLATTN.md -- should
    # appear at the low end first. If motion is genuinely the least-affected
    # axis, 0.3/0.4 should buy back the artifact for less time than lowering
    # tau does. Stills cannot judge this; it needs watching to the end.
    **{f"shipped+start{pct}": (True, dict(SOL_RECOMMENDED_CUDA, start_percent=pct))
       for pct in (0.0, 0.1, 0.3, 0.4)},
}


# Every ratio cheaper than the 16:9/7:4 default, which sits at 1.00x
# attention and otherwise dominates a sweep. Portrait and landscape of a
# ratio cost the same -- packed rows are (h//32)*(w//32), symmetric -- so
# only one orientation of each is here, plus 3:4 because portrait framing is
# a different question from portrait cost.
CHEAP_CANVASES = ("3:2", "4:3", "3:4", "1:1")


def parse_canvas(spec):
    """'4:3' or '1024x768' -> (width, height), always a legal H3 canvas.

    Ratios go through ComfyUI's own `adapt_canvas`, so they land on the 768
    short edge, the 768*1344 area cap and the round-to-32 the model requires
    rather than on whatever the arithmetic produces. Explicit sizes are
    passed through and validated, because the point of naming one is to
    measure that exact canvas.
    """
    from comfy_extras.nodes_minimax_h3 import adapt_canvas

    spec = spec.strip()
    if "x" in spec:
        w, h = (int(v) for v in spec.lower().split("x", 1))
        if w % 32 or h % 32:
            raise SystemExit(f"canvas {spec}: both axes must be multiples of 32")
        return w, h
    if ":" not in spec:
        raise SystemExit(f"canvas {spec!r}: expected 'W:H' or 'WxH'")
    aw, ah = (float(v) for v in spec.split(":", 1))
    if not 0.25 <= aw / ah <= 4:
        # The range the checkpoint was trained over; the reference refuses
        # outside it, so benching there would measure nothing anyone should run.
        raise SystemExit(f"canvas {spec}: aspect {aw / ah:.3g} is outside "
                         f"MiniMax H3's trained 1:4..4:1 range")
    return adapt_canvas(aw, ah)


def canvas_tag(w, h):
    return f"{w}x{h}"


def http_post(url, obj, timeout=60):
    req = urllib.request.Request(
        url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


async def _poll_vram(host, stop, out, period=0.5):
    """Record peak VRAM for a render, two ways, because they are not the same.

    `dev` -- device memory in use, from a single long-lived streaming
    `nvidia-smi`. This is what "peak VRAM" means to a user: how close the box
    came to OOM, allocator pool included.

    `torch` -- ComfyUI's `/system_stats`, in-use as `vram_total - vram_free`.
    That is NOT device usage. `comfy/model_management.py:1785` defines free as
    `mem_free_cuda + (mem_reserved - mem_active)`, i.e. driver-free PLUS
    torch's cached-but-unused reserve, so the difference is torch-ACTIVE
    bytes. Correct for ComfyUI's own model-management decisions, and the right
    number for "how much is really live", but ~12 GB below device usage on
    this box under `--cuda-malloc`.

    Both are reported because reporting one under the name "peak VRAM" is how
    this bench spent 2026-08-14 measuring `reuse_qkv_memory` with an
    instrument that could not see it: every arm came back identical to the
    megabyte, because the active-bytes number was resolving the resident-weight
    plateau rather than the attention transient the flag targets.

    Both are SAMPLED peaks -- a spike shorter than the interval is invisible.
    That is tolerable when the peak is a broad plateau and is why these are
    "peak seen", not "peak". A flag that moves only a brief transient needs a
    different instrument than either of these.
    """
    import aiohttp

    smi = None
    try:
        smi = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits", "-lms", str(int(period * 1000)),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    except Exception:
        pass              # no nvidia-smi: the torch number still gets reported

    async def pump_dev():
        if smi is None or smi.stdout is None:
            return
        while not stop.is_set():
            line = await smi.stdout.readline()
            if not line:
                return
            try:
                out["dev"] = max(out.get("dev", 0), int(line.split()[0]) * 2**20)
            except (ValueError, IndexError):
                pass

    async def pump_torch():
        async with aiohttp.ClientSession() as sess:
            while not stop.is_set():
                try:
                    async with sess.get(f"http://{host}/system_stats",
                                        timeout=aiohttp.ClientTimeout(total=5)) as r:
                        stats = await r.json()
                    for dev in stats.get("devices", []):
                        used = dev.get("vram_total", 0) - dev.get("vram_free", 0)
                        out["torch"] = max(out.get("torch", 0), used)
                except Exception:
                    pass      # a dropped sample is not a failed render
                try:
                    await asyncio.wait_for(stop.wait(), timeout=period)
                except asyncio.TimeoutError:
                    pass

    try:
        await asyncio.gather(pump_dev(), pump_torch())
    finally:
        if smi is not None and smi.returncode is None:
            try:
                smi.kill()
                await smi.wait()
            except Exception:
                pass


async def run_once(host, prompt, client_id, timeout_s, vram=None):
    """Submit and follow the websocket. Returns (total_s, per_node_s, error)."""
    import aiohttp

    stop = asyncio.Event()
    poller = (asyncio.create_task(_poll_vram(host, stop, vram))
              if vram is not None else None)
    try:
        return await _run_once_inner(host, prompt, client_id, timeout_s)
    finally:
        stop.set()
        if poller is not None:
            await poller


async def _run_once_inner(host, prompt, client_id, timeout_s):
    import aiohttp

    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(
            f"ws://{host}/ws?clientId={client_id}", heartbeat=30
        ) as ws:
            t_submit = time.perf_counter()
            resp = http_post(f"http://{host}/prompt",
                             {"prompt": prompt, "client_id": client_id})
            prompt_id = resp["prompt_id"]

            per_node, current, t_node = {}, None, None
            deadline = time.perf_counter() + timeout_s
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None, per_node, f"timed out after {timeout_s:.0f}s"
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
                except asyncio.TimeoutError:
                    return None, per_node, f"timed out after {timeout_s:.0f}s"
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                mtype, d = data.get("type"), data.get("data", {})
                if d.get("prompt_id") not in (None, prompt_id):
                    continue

                if mtype == "executing":
                    now = time.perf_counter()
                    if current is not None and t_node is not None:
                        per_node[current] = per_node.get(current, 0.0) + (now - t_node)
                    node = d.get("node")
                    if node is None:                      # run finished
                        return now - t_submit, per_node, None
                    current, t_node = node, now
                elif mtype == "execution_error":
                    return None, per_node, d.get("exception_message", "execution error")
                elif mtype == "execution_interrupted":
                    return None, per_node, "interrupted"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--runs", type=int, default=3, help="timed runs per arm")
    ap.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    ap.add_argument("--width", type=int, default=DEFAULTS["width"])
    ap.add_argument("--height", type=int, default=DEFAULTS["height"])
    ap.add_argument("--length", type=int, default=DEFAULTS["length"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--arms", default="off,sage",
                    help="comma-separated arms, first is the baseline. "
                         "Known: off, sage, sol, sage+sol, sage+sol+morton")
    ap.add_argument("--canvases", default="",
                    help="comma-separated canvases, crossed with --arms. Either "
                         "a ratio ('1:1', '4:3', '3:4', '3:2') resolved through "
                         "ComfyUI's adapt_canvas so it is always a legal H3 "
                         "canvas, or an explicit '1024x768'. 'cheap' expands to "
                         f"{','.join(CHEAP_CANVASES)} -- every ratio below the "
                         "16:9 default, which at 1.00x attention otherwise "
                         "dominates a sweep's runtime. Empty uses "
                         "--width/--height.")
    ap.add_argument("--vram-arms", default="",
                    help="comma-separated 'headN/ffnM' VRAM-knob settings, "
                         "crossed with --arms. e.g. "
                         "'head1/ffn1,head4/ffn1,head1/ffn2,head4/ffn2' is the "
                         "2x2 that says whether the two knobs are additive. "
                         "Empty leaves both off.")
    ap.add_argument("--video-vae", default=DEFAULTS["video_vae"],
                    help="comma-separated video VAEs. More than one crosses "
                         "them with --arms, so a VAE comparison alternates "
                         "like any other arm instead of running as two "
                         "separate invocations that share no thermal state.")
    ap.add_argument("--skip-warmup", action="store_true",
                    help="Only if a comparable render already ran this session. "
                         "A cold first run pays model load and Triton autotune and "
                         "will read as a large fake win for whichever arm is second.")
    ap.add_argument("--sol-backend", choices=("cuda", "triton"), default="cuda",
                    help="Which Sol-Attn node the sol arms build. cuda "
                         "(SolAttnMiniMax on comfy_kitchen.sol_attn) is the "
                         "default as of 2026-08-14 -- verify the kernel is "
                         "present with bench/check_sol_kernel.py, because a "
                         "stock comfy-kitchen silently falls back to dense. "
                         "triton (SolAttnPatch) is kept for reproducing "
                         "pre-2026-08-14 numbers, which were all taken on it.")
    args = ap.parse_args()

    global SOL_BACKEND
    SOL_BACKEND = args.sol_backend

    cfg = dict(DEFAULTS, steps=args.steps, width=args.width,
               height=args.height, length=args.length)
    client_id = str(uuid.uuid4())
    SAMPLER_NODE = "10"
    DECODE_NODE = "11"   # VAEDecode (video). "12" is VAEDecodeAudio.

    print(f"MiniMax H3 e2e A/B  {cfg['width']}x{cfg['height']} "
          f"length={cfg['length']} steps={cfg['steps']} seed={args.seed}")
    print(f"host={args.host}  runs={args.runs} per arm  (same seed both arms)\n")

    # ComfyUI caches node outputs, so re-submitting an identical graph
    # executes nothing and returns in milliseconds -- which reads as an
    # enormous fake speedup. Each iteration therefore gets its own seed,
    # shared by both arms so the A/B stays paired, and the warmup gets a
    # seed of its own so it cannot alias the first timed run.
    def seed_for(i):
        return args.seed + i

    arms = [a for a in _split_arms(args.arms) if a]
    unknown = [a for a in arms if a not in ARMS and not _is_adhoc(a)]
    if unknown:
        print(f"unknown arm(s) {unknown}; known: {list(ARMS)}")
        print("or ad-hoc: sage+sol[tau=1.6,int8_qk=1] / sol[start_percent=0.1]")
        return 1

    # Named arms are written in the Triton vocabulary. Under --sol-backend
    # cuda a Triton-only knob has nowhere to go, and dropping it silently
    # would turn `sage+sol+int8` into plain `sol` while it still prints as an
    # int8 result -- a whole arm quietly measuring something else. Refuse.
    if SOL_BACKEND != "triton":
        for name in arms:
            if _is_adhoc(name) or name not in ARMS:
                continue
            orphan = sorted(set(ARMS[name][1] or {}) - set(sol_knobs()))
            if orphan:
                print(f"arm {name!r} sets {orphan}, which the {SOL_BACKEND} node "
                      f"does not have.\nIt would silently become a different arm. "
                      f"Drop it, or run --sol-backend triton.")
                return 2

    # A Sol-Attn arm below the length floor cannot produce a signal, and a null
    # there reads as "the knob does nothing" rather than "this run could not
    # have shown anything". docs/SOLATTN.md carries the full note, including
    # that that page's own 124-frame frontier table sits under this floor.
    # Warn rather than gate -- a short run is legitimate for proving the chain
    # composes at all, which is what the verbose arm is for.
    vt = video_tokens(cfg["width"], cfg["height"], cfg["length"])
    if vt < SOL_TOKEN_FLOOR and any(resolve_arm(a)[1] is not None for a in arms):
        print(f"WARNING: {cfg['width']}x{cfg['height']} at length {cfg['length']} is "
              f"{vt:,} video tokens, below the\n"
              f"         ~{SOL_TOKEN_FLOOR:,} floor where Sol-Attn's gains become "
              f"visible. Upstream reports\n"
              f"         nothing measurable under ~250-300 frames at 1344x768. A "
              f"null result here\n"
              f"         is uninformative, not evidence the setting does not matter.\n"
              f"         Raise --length, or the canvas -- tokens are "
              f"latent_t x (w/32) x (h/32),\n"
              f"         so both axes count. 362 frames at 1344x768 is "
              f"{video_tokens(1344, 768, 362):,}, near\n"
              f"         the model's ~100k ceiling.\n")

    # Crossing the VAE axis with the arm axis rather than bolting it on: a
    # VAE swap invalidates MiniMaxH3ImageToVideo (it takes the same VAE for
    # keyframe encoding), so it re-runs the sampler too. Alternating means
    # that shared cost lands on both sides instead of on whichever ran second.
    vaes = [v.strip() for v in args.video_vae.split(",") if v.strip()]

    def vae_tag(name):
        return (name.removeprefix("minimax_h3_video_vae_")
                    .removesuffix(".safetensors"))

    # The VRAM knobs are a third axis, crossed the same way and for the same
    # reason. Both of them shrink transients rather than change arithmetic, so
    # they belong beside the arm rather than inside it.
    def parse_vram_arm(spec):
        head = ffn = 1
        for part in spec.split("/"):
            part = part.strip()
            if part.startswith("head"):
                head = int(part[4:])
            elif part.startswith("ffn"):
                ffn = int(part[3:])
            elif part:
                raise SystemExit(f"unknown --vram-arms token {part!r}; "
                                 f"expected headN or ffnM")
        return head, ffn

    vram_arms = [s.strip() for s in args.vram_arms.split(",") if s.strip()] or [""]
    knobs = [(s, *parse_vram_arm(s)) for s in vram_arms]

    # The canvas is the biggest lever of the lot -- attention is O(S^2) and
    # 1:1 is 0.33x the default's -- so it is an axis, not a global setting.
    if args.canvases.strip():
        specs = [s.strip() for s in args.canvases.split(",") if s.strip()]
        expanded = []
        for s in specs:
            expanded.extend(CHEAP_CANVASES if s == "cheap" else [s])
        canvases = [parse_canvas(s) for s in expanded]
    else:
        canvases = [(cfg["width"], cfg["height"])]

    combos = []
    for cw, ch in canvases:
        for v in vaes:
            for spec, head, ffn in knobs:
                for arm in arms:
                    label = arm
                    if len(canvases) > 1:
                        label += f" {canvas_tag(cw, ch)}"
                    if len(vaes) > 1:
                        label += f"@{vae_tag(v)}"
                    if len(knobs) > 1:
                        label += f" {spec}"
                    combos.append((label, arm, v, head, ffn, cw, ch))
    labels = [c[0] for c in combos]

    def graph_for(combo, seed):
        _label, arm, vae, head, ffn, cw, ch = combo
        use_sage, sol = resolve_arm(arm)
        return build_prompt(dict(cfg, video_vae=vae, width=cw, height=ch),
                            sage=use_sage, seed=seed, sol=sol,
                            head_chunks=head, ffn_chunks=ffn)

    if not args.skip_warmup:
        print("warmup (discarded: model load + Triton autotune + text encode) ...", flush=True)
        total, _, err = asyncio.run(run_once(
            args.host, graph_for(combos[0], seed_for(0)), client_id, args.timeout))
        if err:
            print(f"  warmup FAILED: {err}")
            return 1
        print(f"  {total:.1f}s\n")

    results = {a: [] for a in labels}
    sampler = {a: [] for a in labels}
    decode = {a: [] for a in labels}
    vram = {a: [] for a in labels}
    width = max(len(a) for a in labels)
    for i in range(args.runs):
        seed = seed_for(i + 1)
        for combo in combos:
            label = combo[0]
            peak = {}
            total, per_node, err = asyncio.run(run_once(
                args.host, graph_for(combo, seed), client_id, args.timeout,
                vram=peak))
            if peak.get("dev") or peak.get("torch"):
                vram[label].append((peak.get("dev", 0) / 2**20,
                                    peak.get("torch", 0) / 2**20))
            if err:
                print(f"  run {i+1} {label}: FAILED: {err}")
                return 1
            s = per_node.get(SAMPLER_NODE)
            if s is None:
                print(f"  run {i+1} {label}: sampler node never executed -- ComfyUI "
                      f"served this graph from cache, so there is no timing to "
                      f"report. Vary the seed or restart ComfyUI.")
                return 1
            d = per_node.get(DECODE_NODE)
            results[label].append(total)
            sampler[label].append(s)
            # Absent rather than zero: a cached or skipped decode is not a
            # decode that took no time, and averaging a 0 into it would read
            # as a win. Reported as n/a below if it never ran.
            if d is not None:
                decode[label].append(d)
            pk = vram[label][-1] if vram[label] else None
            print(f"  run {i+1} {label:{width}s}  seed={seed}  total {total:7.1f}s   "
                  f"sampler {s:7.1f}s   decode "
                  f"{f'{d:6.1f}s' if d is not None else '   n/a'}   peak dev "
                  f"{f'{pk[0]:7.0f}' if pk else '    n/a'} / torch "
                  f"{f'{pk[1]:7.0f} MiB' if pk else '    n/a'}", flush=True)

    print()
    med = statistics.median
    base = labels[0]
    b_s, b_t = med(sampler[base]), med(results[base])
    b_d = med(decode[base]) if decode[base] else None
    # Ratios are within-canvas. Comparing a 1:1 arm against a 16:9 baseline
    # would show a ~3x "speedup" that is entirely the canvas and says nothing
    # about the arm -- the exact confound a canvas axis invites.
    base_of = {}
    for label, _arm, _v, _h, _f, cw, ch in combos:
        base_of.setdefault((cw, ch), label)
    canvas_of = {c[0]: (c[5], c[6]) for c in combos}
    multi_canvas = len({(c[5], c[6]) for c in combos}) > 1

    ref_tokens = max((cw // 32) * (ch // 32) for _l, _a, _v, _h, _f, cw, ch in combos)
    # Device peak, not torch-active. See _poll_vram: they differ by ~12 GB
    # under --cuda-malloc, and reporting either as bare "peak VRAM" is what
    # made the reuse_qkv_memory arm unmeasurable on 2026-08-14.
    print(f"{'arm':{width}s} {'sampler':>11s} {'decode':>11s} {'peak dev':>13s} "
          f"{'vs base':>11s} {'sampler vs base':>17s}"
          + (f" {'attn O(S^2)':>12s}" if multi_canvas else ""))
    for label in labels:
        cw, ch = canvas_of[label]
        my_base = base_of[(cw, ch)]
        r_s = med(sampler[my_base])
        r_v = med([v[0] for v in vram[my_base]]) if vram[my_base] else None
        s = med(sampler[label])
        d = med(decode[label]) if decode[label] else None
        pk = med([v[0] for v in vram[label]]) if vram[label] else None
        d_col = f"{d:10.1f}s" if d is not None else f"{'n/a':>11s}"
        v_col = f"{pk:9.0f} MiB" if pk else f"{'n/a':>13s}"
        v_dif = (f"{pk - r_v:+10.0f} " if (pk and r_v) else f"{'n/a':>11s}")
        line = (f"{label:{width}s} {s:10.1f}s {d_col} {v_col} {v_dif} "
                f"{r_s/s:16.3f}x")
        if multi_canvas:
            # What O(S^2) predicts for this canvas. A measured spread that
            # tracks this column is the canvas; one that does not is the
            # part of the step attention never reached.
            line += f" {(((cw // 32) * (ch // 32)) / ref_tokens) ** 2:11.3f}x"
        print(line)
    if multi_canvas:
        print("\n  Ratios are within-canvas -- each canvas is compared to its own "
              "first arm.\n  'attn O(S^2)' is the predicted attention cost "
              "relative to the largest canvas here.")
    # The two links in the hypothesis, kept apart on purpose: a knob can free
    # memory and still return nothing, and reading one number cannot tell that
    # apart from a knob that never freed anything.
    print("\n  'vs base' is MiB freed (negative = less VRAM used).")
    print("  A negative there with 1.000x beside it means the headroom is real "
          "and does not\n  convert -- which is the outcome h3_config.py's ~2.6% "
          "ceiling predicts.")
    print(f"\nsampler share of total on {base}: {100*b_s/b_t:.0f}%  "
          f"-- the ceiling on what any attention work can move")
    if b_d:
        print(f"decode share of total on {base}: {100*b_d/b_t:.0f}%  "
              f"-- the ceiling on what any VAE work can move")
    return 0


if __name__ == "__main__":
    sys.exit(main())
