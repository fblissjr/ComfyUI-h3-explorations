#!/usr/bin/env python3
"""Grade a graph's prompt and price its sequence, BEFORE you press Queue.

    python bench/preflight_graph.py internal/refs/test.json
    python bench/preflight_graph.py workflows/*_api.json workflows/image/*_api.json

**Both directories, and that is not a style choice.** `workflows/*_api.json` is
non-recursive, and the single-frame image graphs live in `workflows/image/`
since 2026-08-16 -- eight of them, all wiring references, including the
three-reference scene that is the most expensive thing on that path. The
documented one-directory invocation silently priced 20 graphs and missed 8,
which is the same shape as the bug that had `check_ref_prompt_labels` exiting 0
over 20 ref graphs instead of 28 that morning: no error, no warning, just a
smaller number nobody had a prior for.

This script takes paths from `sys.argv` and never globs on its own, so it is
explicit rather than blind. If it ever grows a default corpus, route that
through `h3_config.graph_paths()` rather than a literal glob -- the existing
check that graphs are discoverable verifies `GRAPH_DIRS` covers what is on
disk, and does NOT verify that a given walker uses it.

Why this exists. Everything expensive that went wrong on 2026-08-16 was an
unvalidated INPUT, not a bad measurement: a prompt telling the model a mountain
landscape has "architecture", a 15-second clip carrying three seconds of
instruction, a reference budget nobody priced. All of it was decidable before
the render and none of it was decided.

Two surfaces already exist and neither closes this:

  MiniMaxH3Resolution  computes `video_tokens` and three siblings, and those
                       outputs are wired to NOTHING in all 40 shipped graphs.
                       It also only knows canvas x length -- it cannot see
                       references, which is where the cost actually is.
  MiniMaxH3Preflight   counts the real packed layout including references, but
                       it is a node: it runs at execution, after you have
                       committed, and it only prints. Better than nothing (the
                       number lands seconds in, not 40 minutes in) and still
                       not "know before you run".

So this is static. No CUDA, no model, no server. Reads the graph, resolves the
reference images on disk, and reports.

**It reports; it does not refuse.** A tool that blocks you in your own
tinkering repo gets disabled, and then it protects nothing. Exit is 0 unless
the file cannot be read.

## What the numbers are worth

Token counts are EXACT where they are derived and marked `~` where they are
not. Video and image-reference rows are arithmetic on the graph, so they are
exact. The text segment is measured with ComfyUI's own tokenizer when it can be
imported, and the conditioner's vision blocks are estimated from the measured
rule that text lands 75-160 rows above the reference segment
(`docs/h3_references.md`), because computing them properly needs the vision
encoder.

**Peak VRAM is deliberately NOT predicted.** Two datapoints on this box:
78,019 tokens peaked at 21,938 MiB, and 124k tokens peaked at 17,840 MiB --
more tokens, lower peak. `h3_config.py` records why: process peak tracks
ComfyUI's dynamic allocator against free memory, not what the model holds. A
formula here would be wrong in both directions, so the report prints the
recorded datapoints and lets you judge.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "bench"))

import h3_rules  # noqa: E402
import vendor_config  # noqa: E402

# The ordinal rule lives in the label check and must not have a second copy
# here -- a soundtrack's <Audio j> is emitted before its own <Video k>, and a
# duplicated version of that would drift silently.
from check_ref_prompt_labels import wired_labels  # noqa: E402

# Both waivers are imported, never restated. `_audio_sections_optional` reads
# the GRAPH for an audio decoder rather than matching a name, and
# `_STRUCTURE_PROBES` names the two graphs that are unstructured on purpose. A
# second copy of either would drift, and the first run of this file without
# them reported FAIL on 7 of 8 image graphs for sections that structurally
# cannot apply -- a checker going red while the state is correct, which is the
# one thing docs/checks.md says is worse than no checker.
from check_prompt_guide_conformance import (  # noqa: E402
    BASE_GUIDE, _STRUCTURE_PROBES, _audio_sections_optional)


def _core_minimax_cpu():
    """Import the installed H3 arithmetic with Comfy forced to CPU mode.

    Preflight is explicitly a no-GPU tool. Merely importing Comfy's model
    management otherwise selects the CUDA device on this install, even though
    the only functions read here are scalar geometry helpers.
    """
    import importlib
    comfy_root = str(Path.home() / "ComfyUI")
    if sys.path[0] != comfy_root:
        sys.path.insert(0, comfy_root)
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    return importlib.import_module("comfy_extras.nodes_minimax_h3")

# Every node that carries a prompt into the encoder. `MiniMaxH3Conditioning`
# is this repo's own, and the fl2va path moved onto it on 2026-08-21; a file
# that knows only the first name reports "nothing to grade" over every graph
# this repo now ships and reads as a clean pass.
PROMPT_NODES = ("MiniMaxH3ReferenceToVideo", "MiniMaxH3Conditioning",
                "MiniMaxH3ReferenceConditioning")
FIT_NODE = "MiniMaxH3ReferenceFit"
LOAD_IMAGE = "LoadImage"

# THE RELEASE SHIPS TWO PROMPT GUIDES AND THEY DO NOT SHARE A SECTION LIST.
# Grading every prompt against the six-section one is how a correct t2v prompt
# came back `sections absent: subject_definitions, summary, retention_analysis,
# detailed_description` -- four failures for obeying the guide that applies.
#
#   base-en.txt:39-48  T2VA / I2VA / FL2VA / L2VA -- "the three core fields"
#   ref-en.txt:12-22   full-reference mode -- "six sections in the following
#                      order"
#
# Which one applies is READ OFF THE GRAPH, never off a filename: full-reference
# mode is the mode that wires reference labels, so a graph with any `ref_*`
# socket is graded against ref-en and everything else against base-en. That
# derivation was checked against every shipped graph before it was written
# here, and disagreed with none of them.
BASE_SECTIONS = ["integrated_multimodal_description", "overall_soundscape",
                 "non_diegetic_music"]
REF_SECTIONS = ["subject_definitions", "summary", "retention_analysis",
                "detailed_description", "overall_soundscape",
                "non_diegetic_music"]

# The field the shot, cut-timing and dialogue rules read. ref-en.txt:229 names
# the pair outright: "Main field | integrated_multimodal_description |
# detailed_description".
MAIN_FIELD = {"base": "integrated_multimodal_description",
              "ref": "detailed_description"}

REF_SOCKET_PREFIXES = ("ref_images.", "ref_videos.", "ref_audios.",
                       "ref_video_audios.")
KEYFRAME_SOCKETS = ("first_frame", "last_frame")


def _base_alignment_templates() -> dict[str, str]:
    """The three keyframe instructions, parsed from the release guide.

    The strings differ in punctuation and bracket use as well as wording. A
    local copy would turn the check into a comparison against ourselves, so
    the guide is the baseline and a missing/unparseable guide is a loud error.
    """
    text = BASE_GUIDE.read_text()
    out = {}
    for mode in ("I2VA", "FL2VA", "L2VA"):
        match = re.search(
            rf"\*\*{mode}\*\* always uses:\s*```text\s*(.*?)\s*```",
            text, re.S)
        if match is None:
            raise RuntimeError(
                f"could not parse {mode}'s alignment instruction from "
                f"{BASE_GUIDE}")
        out[mode] = match.group(1).strip()
    return out


BASE_ALIGNMENT = _base_alignment_templates()

# THE FIVE MARKERS NEITHER OFFICIAL GUIDE DOCUMENTS.
#
# The release declares seven special tokens. Both prompt guides describe only
# `<d>` / `</d>`; `<|cutoff|>`, `<|lyrics_start|>`, `<|lyrics_end|>`,
# `<|caption_start|>` and `<|caption_end|>` appear in neither. So the rules
# below cannot be parsed out of the guide the way
# `check_prompt_guide_conformance.py` parses its vocabulary -- they are house
# pattern, taken from the worked examples in `workflows/build_workflows.py`
# (`T2V_SCENES`, `REF_SCENE_SHOTS`) and `bench/audit_h3_marker_tokenization.py`.
# That is also why they live here and not in the conformance checker: that file
# refuses to assert anything the guide does not state, and it is right to.
#
# The escaped instance, which is what earns a new check at all
# (CLAUDE.md: cite one before building). On 2026-08-22 a prompt written
# elsewhere came through with three caption pairs on their own lines, one
# padded with spaces, one spoken line split across two adjacent pairs, and a
# trailing space inside a `<d>`. Run through this file as it stood, it scored
# exactly one WARN, for word count. Every marker defect was invisible, because
# nothing here looked at a marker other than `<d>`.
#
# The nesting and balance cases below did NOT escape -- no instance of either
# has been seen. They are here because the same day
# `internal/PROMPTING.md` gained a "must" stating the nesting rule, and
# CLAUDE.md's standing rule is that a "must" with no assertion behind it is an
# uncontrolled requirement. This is that assertion.
#
# What these deliberately do NOT decide: whether a caption is on the RIGHT
# content. A caption is burned-in on-screen text, so its string may legitimately
# differ from the dialogue -- a subtitle is the case in point -- and no
# mechanical rule can tell an intended subtitle from a stray transcript. Only
# reading it finds that.
MARKER_PAIRS = (("<|lyrics_start|>", "<|lyrics_end|>"),
                ("<|caption_start|>", "<|caption_end|>"))
LYRICS, CAPTION = MARKER_PAIRS
ALL_MARKERS = tuple(m for pair in MARKER_PAIRS for m in pair) + ("<|cutoff|>",)


def marker_rules(prompt: str, main_body: str) -> list[tuple[str, str]]:
    """Structure of the five undocumented markers. Decidable from the text."""
    out = []
    for op, cl in MARKER_PAIRS:
        if prompt.count(op) != prompt.count(cl):
            out.append(("FAIL", f"{op} appears {prompt.count(op)}x against "
                                f"{prompt.count(cl)}x for {cl}"))
    for op, cl in MARKER_PAIRS:
        for m in re.finditer(re.escape(op) + r"(.*?)" + re.escape(cl),
                             prompt, re.S):
            body = m.group(1)
            has_d = "<d>" in body or "</d>" in body
            if has_d and (op, cl) == CAPTION:
                out.append(("FAIL", "a <d> block sits inside a caption pair; "
                                    "captions are a SIBLING of <d>, never a "
                                    "wrapper. Only the lyrics pair wraps <d>"))
            if not has_d and (op, cl) == LYRICS:
                out.append(("FAIL", "a lyrics pair wraps no <d> block, so it "
                                    "marks nothing as sung"))
            if any(o in body for o, _ in MARKER_PAIRS):
                out.append(("FAIL", f"{op} contains another marker pair"))
            if (op, cl) == CAPTION and body != body.strip():
                out.append(("WARN", f"caption content {body!r} is padded with "
                                    f"whitespace; the padding is part of the "
                                    f"string"))
    for m in re.finditer(r"<d>(.*?)</d>", prompt, re.S):
        if any(x in m.group(1) for x in ALL_MARKERS):
            out.append(("FAIL", "a marker pair sits inside a <d> block; <d> "
                                "carries a language tag and the words, and "
                                "nothing else"))
        if m.group(1) != m.group(1).rstrip():
            out.append(("WARN", "a <d> block has whitespace before </d>; "
                                "MiniMax's own examples close tight against "
                                "the last character"))
    # Only on an UNPATCHED tokenizer, which is the state to assume: a pack
    # cannot know whether the install carries the special tokens. There, BPE
    # pulls the full stop into the marker's leading fragment, so the marker
    # retokenizes the sentence before it. The house dialogue scenes sit the
    # marker against `</d>` with no stop at all. Not a FAIL: the audit
    # harness's non-dialogue `hard_cut` scene legitimately has a stop there.
    for m in re.finditer(r"(.)<\|cutoff\|>", prompt):
        if m.group(1) == ".":
            out.append(("WARN", "a full stop sits directly before <|cutoff|>"))
    for line in main_body.splitlines():
        stripped = line.lstrip()
        for mk in ALL_MARKERS + ("<d>",):
            if stripped.startswith(mk):
                out.append(("WARN", f"a line opens with {mk}; markers ride "
                                    f"inline in the shot prose, and only "
                                    f"[Shot N] starts a line"))
                break
    for _ in re.finditer(re.escape(CAPTION[1]) + r"\s*" + re.escape(CAPTION[0]),
                         prompt):
        out.append(("WARN", "two caption pairs are adjacent with only "
                            "whitespace between them; one on-screen line is "
                            "one pair"))
    return out


def guide_for(inputs: dict) -> str:
    """"ref" when the node wires reference labels, else "base"."""
    return "ref" if ("references" in inputs or
                     any(k.startswith(REF_SOCKET_PREFIXES) for k in inputs)) \
        else "base"


def _reference_media(inputs: dict, graph: dict):
    """Socket-shaped media view plus per-image policy for either surface.

    The typed chain remains the source of order and ownership; this adapter is
    only for older pricing/reporting code whose natural unit is a media link.
    Chain traversal comes from ``reference_order`` so a malformed chain cannot
    be priced as a shorter, apparently valid one.
    """
    if "references" not in inputs:
        return inputs, {}, False
    from reference_order import resolve_chain_entries
    link = inputs.get("references")
    if not (isinstance(link, list) and len(link) == 2):
        # `wired_labels` will report the precise chain error during grading.
        return {}, {}, True
    media, image_policies = {}, {}
    counts = {"image": 0, "video": 0, "audio": 0}
    for _nid, append, kind in resolve_chain_entries(graph, str(link[0])):
        append_inputs = append.get("inputs", {})
        index = counts[kind]
        counts[kind] += 1
        if kind == "image":
            key = f"ref_images.ref_image_{index}"
            media[key] = append_inputs.get("image")
            image_policies[key] = append_inputs.get("size_policy", "match")
        elif kind == "video":
            media[f"ref_videos.ref_video_{index}"] = append_inputs.get("frames")
            soundtrack = append_inputs.get("soundtrack")
            if soundtrack is not None:
                media[f"ref_video_audios.ref_video_audio_{index}"] = soundtrack
        else:
            media[f"ref_audios.ref_audio_{index}"] = append_inputs.get("audio")
    return media, image_policies, True

VISUAL_MARKERS = {"fully_preserved", "partially_preserved",
                  "attribute_transfer", "weak_reference"}
AUDIO_MARKERS = {"fully_copy", "partially_copy", "reference", "weak_reference"}

# Where ComfyUI actually looks for reference images. Read from start.sh rather
# than assumed: this install runs --input-directory on a share, so the stock
# ComfyUI/input path is empty and a naive resolver reports every reference
# missing.
def _input_dirs() -> list[Path]:
    dirs = []
    start = Path.home() / "ComfyUI" / "start.sh"
    if start.exists():
        # findall, not search, and require a path: the flag is documented in a
        # comment block above the real invocation, so the first match is the
        # word "where" from the help text.
        for cand in re.findall(r"--input-directory\s+(\S+)", start.read_text()):
            if cand.startswith(("/", "~", "$")):
                dirs.append(Path(cand).expanduser())
    dirs.append(Path.home() / "ComfyUI" / "input")
    return dirs


def resolve_image(name: str) -> Path | None:
    for d in _input_dirs():
        p = d / name
        if p.exists():
            return p
    return None


# ComfyUI resolves `embedding:name` in an H3 prompt through the ordinary
# SD1Tokenizer path (`MiniMaxH3Tokenizer` passes `embedding_size=5120,
# embedding_key="qwen3vl_32b"`, merged upstream in PR #15697). What it does
# with one it cannot find is the reason this grader exists:
# `comfy/sd1_clip.py` logs `warning, embedding:<name> does not exist, ignoring`
# and renders anyway. The render completes, the prompt is quietly missing a
# concept, and nothing in a queued job's output says so.
H3_EMBEDDING_KEY = "qwen3vl_32b"
# The DiffSynth exports name their tensor `weight`; core reads that shape too
# (`bundled_embed` / the `embed_key` fallback), so a file carrying it is
# loadable and must not be reported as broken.
H3_EMBEDDING_ALT_KEY = "weight"
H3_HIDDEN_SIZE = 5120
_EMBEDDING_REF = re.compile(r"embedding:([^\s,.;:!?)\]}\"']+)")


def _embedding_dirs() -> list[Path]:
    """Where ComfyUI would look, deduped and order-preserving.

    `folder_paths` is the authority (it honours extra_model_paths.yaml); the
    literal is the fallback for running this with no ComfyUI on the path, which
    is the state most of this file already tolerates.
    """
    dirs = []
    try:
        sys.path.insert(0, str(Path.home() / "ComfyUI"))
        import folder_paths
        dirs += [Path(d) for d in folder_paths.get_folder_paths("embeddings")]
    except Exception:
        pass
    dirs.append(Path.home() / "ComfyUI" / "models" / "embeddings")
    seen, out = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def resolve_embedding(name: str) -> tuple[str, str]:
    """(state, detail) for one `embedding:` reference. Never raises.

    States are `ok`, `missing`, and `unreadable` -- three, not two, because a
    file that is present but whose header cannot be parsed is a different
    problem from one that is absent, and reporting either as the other sends
    the reader to the wrong place. Core accepts several extensions; the
    reference may also carry a subfolder (`embedding:h3/name`).
    """
    stem = name.strip()
    cands = []
    for d in _embedding_dirs():
        for ext in ("", ".safetensors", ".pt", ".bin"):
            cands.append(d / f"{stem}{ext}")
    hit = next((p for p in cands if p.is_file()), None)
    if hit is None:
        return "missing", (
            f"no file for `embedding:{stem}` under "
            f"{', '.join(str(d) for d in _embedding_dirs())}")
    if hit.suffix != ".safetensors":
        # Only the safetensors header is cheap to read without torch.
        return "unreadable", f"{hit.name} is not safetensors; header not read"
    try:
        from safetensors import safe_open
        with safe_open(hit, framework="pt") as f:
            keys = list(f.keys())
            key = next((k for k in (H3_EMBEDDING_KEY, H3_EMBEDDING_ALT_KEY)
                        if k in keys), None)
            if key is None:
                return "unreadable", (
                    f"{hit.name} holds {keys}, neither "
                    f"`{H3_EMBEDDING_KEY}` nor `{H3_EMBEDDING_ALT_KEY}`")
            shape = list(f.get_slice(key).get_shape())
    except Exception as exc:
        return "unreadable", f"{hit.name}: {type(exc).__name__}: {exc}"
    if len(shape) != 2 or shape[-1] != H3_HIDDEN_SIZE:
        return "unreadable", (
            f"{hit.name} key `{key}` is {shape}, not "
            f"[tokens, {H3_HIDDEN_SIZE}]")
    return "ok", f"{hit.name} `{key}` {shape[0]} token(s)"


def embedding_notes(prompt: str) -> list[tuple[str, str]]:
    """Findings for every `embedding:` reference in a prompt."""
    out = []
    for name in _EMBEDDING_REF.findall(prompt or ""):
        state, detail = resolve_embedding(name)
        if state == "ok":
            out.append(("note", f"embedding:{name} resolves -- {detail}"))
        elif state == "missing":
            # WARN, not FAIL: core does not refuse either, and this file
            # reports rather than blocking. The point is that the render will
            # silently proceed WITHOUT it.
            out.append(("WARN", f"embedding:{name} does not resolve, and "
                                f"ComfyUI will render without it rather than "
                                f"refuse -- {detail}"))
        else:
            out.append(("WARN", f"embedding:{name} present but not gradeable "
                                f"-- {detail}"))
    return out


def embedding_tokens(prompt: str) -> tuple[int, int]:
    """(rows contributed by resolved embeddings, count that did NOT resolve).

    The second half is why this returns a pair: an unresolved reference costs
    zero rows AND drops a concept, and a price that quietly omits both reads as
    a graph that is cheaper than it is.
    """
    rows = unresolved = 0
    for name in _EMBEDDING_REF.findall(prompt or ""):
        state, detail = resolve_embedding(name)
        if state != "ok":
            unresolved += 1
            continue
        m = re.search(r"(\d+) token", detail)
        if m:
            rows += int(m.group(1))
    return rows, unresolved


def _audio_source(graph: dict, val) -> tuple[dict | None, float | None]:
    """Walk back from a ref-audio socket to the node holding the media.

    Returns (source node, trim duration in seconds). The trim is `None` when
    the socket is fed directly, which is the state this whole grader exists
    to report.
    """
    trim = None
    seen = set()
    while isinstance(val, list) and val and str(val[0]) in graph:
        nid = str(val[0])
        if nid in seen:
            return None, trim          # a cycle cannot ship, but do not hang on one
        seen.add(nid)
        node = graph[nid]
        if node.get("class_type") == "TrimAudioDuration":
            trim = node["inputs"].get("duration")
            val = node["inputs"].get("audio")
            continue
        return node, trim
    return None, trim


def audio_channels(path: Path) -> int | None:
    """Channel count of a media file's first audio stream, via ffprobe.

    Returns None when ffprobe is absent or the file has no audio stream --
    both of which are "cannot tell", not "stereo". The caller must not treat
    a None as a pass.
    """
    import shutil
    import subprocess
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channels", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    first = out.stdout.strip().splitlines()
    try:
        return int(first[0])
    except (IndexError, ValueError):
        return None


def video_probe(path: Path) -> tuple[int, int] | None:
    """(width, height) of a media file's first video stream, via ffprobe."""
    import shutil
    import subprocess
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
            capture_output=True, text=True, timeout=20)
        w, h = out.stdout.strip().splitlines()[0].split("x")
        return int(w), int(h)
    except Exception:
        return None


def _release_qwen_grid(n_raw: int, w: int, h: int):
    """The release's Qwen video grid, from the release processor itself.

    **Executed, never reimplemented.** `smart_resize` was hand-modelled twice
    on 2026-08-22 and was wrong both times, so this runs the real
    `Qwen3VLVideoProcessor` against `vendor_config/video_preprocessor_config.json`
    and reads `video_grid_thw` off it. Returns None when transformers or the
    config cannot be reached -- the caller then prints "not calculated" rather
    than substituting local arithmetic, which is the whole point.

    **`n_raw` is the RAW sampled count, not the repeat-padded one**, and the
    difference is not cosmetic: at 1344x768, 31 gives [16,42,74] (1184x672)
    and 32 gives [16,40,72] (1152x640). Both are 16 temporal blocks; the
    pixel budget divides by the frame count it was handed. ComfyUI's pad-to-
    even happens after sampling and the release never sees it, so passing the
    padded count reproduces a number that is wrong by 8%.
    """
    try:
        import torch
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
            Qwen3VLVideoProcessor,
        )
        cfg = json.loads((_REPO / "vendor_config"
                          / "video_preprocessor_config.json").read_text())
        proc = Qwen3VLVideoProcessor(
            **{k: v for k, v in cfg.items()
               if k not in ("processor_class", "video_processor_type")})
        out = proc(videos=[torch.zeros(n_raw, 3, h, w, dtype=torch.uint8)],
                   do_sample_frames=False, input_data_format="channels_first",
                   return_tensors="pt")
        g = out["video_grid_thw"][0].tolist()
        merge = int(proc.merge_size) ** 2
        return g, g[1] * g[2] // merge
    except Exception:
        return None


def _comfy_pair_grid(w: int, h: int, max_pixels: int = 12845056):
    """Rows per two-frame block under ComfyUI's per-pair policy.

    Mirrors `process_video_block` (`comfy/text_encoders/minimax.py:35`): round
    to the patch*merge factor, clamp only if over its own per-PAIR max_pixels,
    then merge 2x2. No clip-wide budget exists on this path.
    """
    factor = 16 * 2
    hb = round(h / factor) * factor
    wb = round(w / factor) * factor
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    return (wb, hb), (hb // 16) * (wb // 16) // 4


def reference_video_report(
    ins: dict, graph: dict, length, video_policy: str = "native-comfy"
) -> list[str]:
    """Both towers, both policies, per reference video.

    **Why both stages.** A reference video is sized twice and the two sizings
    diverge from the release independently: ComfyUI declines to upscale it for
    the VAE (gap 6), and hands Qwen the same array per two-frame pair where
    the release runs a clip-wide pixel budget over the sampled view. Reporting
    one without the other invites closing the upscale alone, which OVERSHOOTS
    the release rather than matching it -- the `gap-6 only` row exists to make
    that visible before anyone builds it.

    Reports, never refuses, and never guesses: an unreachable source or
    processor prints what could not be computed.
    """
    out = []
    for key, val in sorted(ins.items()):
        if not key.startswith("ref_videos.") or val is None:
            continue
        label = key.split(".")[-1]
        src = graph[str(val[0])] if isinstance(val, list) else None
        name = (src or {}).get("inputs", {}).get("video")
        path = resolve_image(name) if isinstance(name, str) else None
        dims = video_probe(path) if path else None
        if dims is None:
            out.append(f"  {label}: source {name!r} not resolved on this box, "
                       f"so neither tower's geometry was calculated")
            continue
        adapt_canvas = _core_minimax_cpu().adapt_canvas
        w, h = dims
        cw, ch = adapt_canvas(w, h)
        comfy_vae = (cw, ch)
        if w * h < cw * ch:                       # gap 6: never upscales
            comfy_vae = (max(32, round(w / 32) * 32), max(32, round(h / 32) * 32))
        rel_vae = (cw, ch)

        n = length if isinstance(length, int) else 0
        while n % 17 != 5 and n > 5:
            n -= 1
        raw = len(range(0, n, 12))
        padded = raw + (raw % 2)
        blocks = padded // 2

        c_grid, c_per = _comfy_pair_grid(*comfy_vae)
        hyb_grid, hyb_per = _comfy_pair_grid(*rel_vae)
        rel = _release_qwen_grid(raw, *rel_vae)

        out.append(f"  {label}: {name}")
        out.append(f"      source                {w}x{h}")
        comfy_active = video_policy in ("comfy", "native-comfy")
        active_kind = ("native core" if video_policy == "native-comfy"
                       else "local typed policy")
        out.append(f"      VAE-prepared, comfy   {comfy_vae[0]}x{comfy_vae[1]}"
                   f"{'   (gap 6: not upscaled)' if comfy_vae != rel_vae else ''}"
                   f"{'   <- ACTIVE (' + active_kind + ')' if comfy_active else ''}")
        out.append(f"      VAE-prepared, release {rel_vae[0]}x{rel_vae[1]}"
                   f"{'   <- ACTIVE (local typed policy)' if video_policy == 'release' else ''}")
        out.append(f"      sampled at 2 fps      {raw} raw -> {padded} emitted "
                   f"({blocks} temporal blocks) from {n} frames")
        out.append(f"      Qwen, comfy           {c_grid[0]}x{c_grid[1]}  "
                   f"{c_per * blocks:>7,} rows"
                   f"{'   <- ACTIVE (' + active_kind + ')' if comfy_active else ''}")
        if rel is None:
            out.append(f"      Qwen, release         NOT CALCULATED -- the release "
                       f"processor or its config could not be reached. No local "
                       f"substitute is offered; the arithmetic was wrong twice.")
        else:
            g, per = rel
            out.append(f"      Qwen, release         {g[2] * 16}x{g[1] * 16}  "
                       f"{per * g[0]:>7,} rows"
                       f"{'   <- ACTIVE (local typed policy)' if video_policy == 'release' else ''}")
        out.append(f"      Qwen, gap-6 only      {hyb_grid[0]}x{hyb_grid[1]}  "
                   f"{hyb_per * blocks:>7,} rows   <- upscale WITHOUT the "
                   f"clip-wide budget")
    return out


def audio_reference_notes(ins: dict, graph: dict, length,
                          typed_boundary: bool = False) -> list[str]:
    """Grade every wired reference-audio socket: trimmed, matched, stereo.

    **Three separate failures, and only one of them is ours.**

    1. *Untrimmed.* The reference pipeline caps every reference soundtrack at
       the generated duration -- sglang computes `frame_count / fps` into
       `ffmpeg -t`, diffusers does the same, for a video's soundtrack and a
       standalone audio reference alike. `_encode_ref_audio`
       (`comfy_extras/nodes_minimax_h3.py:71`) truncates neither, at 80 rows
       per second of excess attended on every sampling step. Shipped graphs
       wire `TrimAudioDuration`; a hand-built one is who this case is for.
    2. *Trim disagrees with the render.* The duration is a baked widget and
       `length` can be patched at submit time without it -- which
       `bench/run_graph_arms.py --set` does routinely. **This case is the
       control named in `ref_audio_seconds`'s docstring**, and without it
       that "must" would be enforced by nothing.
    3. *Mono.* `_encode_ref_audio` does not upmix, so a mono waveform
       produces half the rows the packed layout allocated and the assignment
       raises (`comfy/ldm/minimax/model.py:659`, gap 7). diffusers and
       DiffSynth-Studio expand to stereo first. **Reported, not fixed**: the
       upmix belongs in core's encoder, and wiring one here would alter every
       stereo source to prevent a crash none of them hit.

    Reports, never refuses -- the same contract as the rest of this file.
    """
    out = []
    fps = 24.0
    want = (length / fps) if isinstance(length, (int, float)) else None
    for key, val in sorted(ins.items()):
        if not (key.startswith("ref_video_audios.")
                or key.startswith("ref_audios.")) or val is None:
            continue
        label = key.split(".")[-1]
        src, trim = _audio_source(graph, val)
        if typed_boundary:
            duration = f"{want:.2f}s" if want is not None else "the aligned target"
            out.append(
                f"  {label}: typed boundary caps audio to {duration} and "
                f"normalizes mono to stereo before the audio VAE")
        elif trim is None:
            out.append(f"  {label}: WARN reaches the node untrimmed. The "
                       f"reference pipeline caps every reference soundtrack "
                       f"at the generated duration and ComfyUI caps none, at "
                       f"80 rows per second of excess on every step. Wire "
                       f"TrimAudioDuration.")
        elif want is not None and abs(trim - want) > 0.01:
            out.append(f"  {label}: WARN trimmed to {trim:.2f}s but the render "
                       f"is {want:.2f}s ({length} frames at {fps:g} fps). The "
                       f"trim is a baked widget; patching `length` alone "
                       f"leaves it stale.")
        else:
            out.append(f"  {label}: trimmed to {trim:.2f}s, matching the render")

        # The media itself, when it can be reached.
        name = None
        if src is not None:
            si = src.get("inputs", {})
            name = si.get("audio") or si.get("video")
        path = resolve_image(name) if isinstance(name, str) else None
        if path is None:
            out.append(f"  {label}: source not resolved on this box, so its "
                       f"channel count and length were NOT checked")
            continue
        ch = audio_channels(path)
        if ch is None:
            out.append(f"  {label}: {name} -- channel count unreadable "
                       f"(no ffprobe, or no audio stream). NOT a pass.")
        elif ch == 1 and typed_boundary:
            out.append(f"  {label}: {name} is mono; the typed boundary will "
                       f"duplicate it to stereo")
        elif ch == 1:
            out.append(f"  {label}: WARN {name} is MONO. "
                       f"`_encode_ref_audio` does not upmix, so the packed "
                       f"assignment raises rather than degrading (gap 7). "
                       f"Convert to stereo before queueing.")
        elif ch > 2 and typed_boundary:
            out.append(f"  {label}: FAIL {name} has {ch} channels; the typed "
                       f"boundary refuses to guess which stereo pair to use")
        else:
            out.append(f"  {label}: {name} is {ch}-channel")
    return out


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def fit(w: int, h: int, scale: float, mult: int = 32) -> tuple[int, int]:
    return (max(mult, round(w * scale / mult) * mult),
            max(mult, round(h * scale / mult) * mult))


def rows(w: int, h: int) -> int:
    """DiT rows for a w x h pixel area: VAE /16, then the DiT's 2x2 patch."""
    return (w // 32) * (h // 32)


def split_sections(prompt: str, sections: list[str]) -> dict[str, str]:
    idx = []
    for name in sections:
        m = re.search(rf"^{re.escape(name)}:", prompt, re.M)
        if m:
            idx.append((m.start(), m.end(), name))
    idx.sort()
    out = {}
    for i, (_s, e, name) in enumerate(idx):
        end = idx[i + 1][0] if i + 1 < len(idx) else len(prompt)
        out[name] = prompt[e:end].strip()
    return out


def text_tokens(prompt: str) -> tuple[int, str]:
    """Token count from ComfyUI's own tokenizer, or an estimate if absent."""
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(
            Path.home() / "ComfyUI" / "comfy" / "text_encoders" / "qwen25_tokenizer")
        return len(tk(prompt)["input_ids"]), "exact"
    except Exception:
        return int(len(prompt.split()) * 1.35), "~est"


def _single_frame(node: dict, graph: dict) -> bool:
    """True when this graph renders one frame. Read off the graph, never a name.

    `length` is usually LINKED (image graphs carry `['27', 2]`), so a literal
    read misses every one of them.
    """
    val = node["inputs"].get("length")
    if isinstance(val, int):
        return val == 1
    if isinstance(val, list) and val and str(val[0]) in graph:
        return graph[str(val[0])].get("inputs", {}).get("length") == 1
    return False


def _resolved_length(node: dict, graph: dict) -> int | None:
    """Follow the conditioner's length link to a literal, when observable."""
    value = node.get("inputs", {}).get("length")
    seen = set()
    while isinstance(value, list) and len(value) == 2:
        source_id = str(value[0])
        if source_id in seen or source_id not in graph:
            return None
        seen.add(source_id)
        value = graph[source_id].get("inputs", {}).get("length")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return h3_rules.snap_length(value)


def _expected_base_alignment(node: dict, graph: dict,
                             shots: list[tuple[str, str]]) -> tuple[str | None, str]:
    """Expected Part One from sockets, final shot, and effective duration."""
    ins = node.get("inputs", {})
    first = ins.get("first_frame") is not None
    last = ins.get("last_frame") is not None
    if not first and not last:
        return "", "T2VA"
    if first and not last:
        return BASE_ALIGNMENT["I2VA"], "I2VA"

    mode = "FL2VA" if first else "L2VA"
    if not shots:
        return None, f"{mode}: no [Shot N] exists to resolve the final shot"
    length = _resolved_length(node, graph)
    if length is None:
        return None, f"{mode}: graph length could not be resolved"
    final_shot = int(shots[-1][0])
    seconds = h3_rules.duration_of(length)
    expected = (BASE_ALIGNMENT[mode]
                .replace("Shot N", f"Shot {final_shot}")
                .replace("S.SS", f"{seconds:.2f}"))
    return expected, mode


def grade(node: dict, graph: dict, stem: str = "") -> list[tuple[str, str]]:
    """Mechanical rules only. Every one is decidable from the prompt + sockets."""
    out = []
    ins = node["inputs"]
    prompt = ins.get("prompt", "")
    guide = guide_for(ins)
    sections = REF_SECTIONS if guide == "ref" else BASE_SECTIONS
    main_field = MAIN_FIELD[guide]
    sec = split_sections(prompt, sections)
    # the graph rides along: an ordered graph keeps its plan in the append
    # chain, where this node's inputs cannot see it
    expected = wired_labels(ins, graph)
    if guide == "base":
        out.append(("note", "graded against base-en (T2VA/I2VA/FL2VA/L2VA): "
                            "three core fields, not ref-en's six"))

    required = list(sections)
    if _audio_sections_optional(graph):
        required = [s for s in required
                    if s not in ("overall_soundscape", "non_diegetic_music")]
        out.append(("note", "no VAEDecodeAudio: the two audio sections are "
                            "not required of this graph"))
    if stem in _STRUCTURE_PROBES:
        out.append(("note", f"{stem} is a structure probe, unstructured on "
                            f"purpose: section rules waived, everything else "
                            f"still graded"))
        required = []
    missing = [s for s in required if s not in sec]
    if missing:
        out.append(("FAIL", f"sections absent: {', '.join(missing)}"))
    else:
        order = [s for s in sections if s in sec]
        pos = [prompt.index(s + ":") for s in order]
        if pos != sorted(pos):
            out.append(("FAIL", "sections are out of the guide's order"))

    named = set(re.findall(r"<(?:Picture|Video|Audio) \d+>", prompt))

    # FL2VA NAMES ITS PICTURES WITHOUT BRACKETS, AND THE GUIDE IS EXPLICIT.
    # `base_en.md:14-32` gives one alignment sentence per task and FL2VA's is
    # the only one of the three that carries no angle brackets and no square
    # brackets: "Picture 1 (from Shot 1) ... Picture 2 (from Shot N)". I2VA and
    # L2VA both bracket, so a rule that demands `<Picture N>` is correct for
    # every keyframe graph EXCEPT the two-frame one.
    #
    # This is the one-implementation trap from CLAUDE.md, caught by a second
    # implementation rather than by reasoning: the rule was written when i2v was
    # the only keyframe graph in the repo, it was right about that graph, and it
    # would have failed the first correct fl2va prompt anybody wrote -- pushing
    # them to bracket the labels and diverge from the string the guide says the
    # mode "always uses".
    #
    # Deliberately narrow. The bare form satisfies the requirement ONLY on a
    # graph wiring BOTH keyframe sockets, and only for `Picture`; `named` is
    # unchanged, so the reverse check below still fails a prompt that mentions a
    # label no socket wires.
    satisfied = set(named)
    if all(ins.get(k) is not None for k in KEYFRAME_SOCKETS):
        satisfied |= {f"<Picture {n}>"
                      for n in re.findall(r"(?<![<\w])Picture (\d+)", prompt)}
    for lab in expected:
        if lab not in satisfied:
            out.append(("FAIL", f"{lab} is wired but the prompt never names it"))
    for lab in sorted(named):
        if lab not in expected:
            out.append(("FAIL", f"prompt names {lab}, which no socket wires"))

    defined = re.findall(r"^(<Subject \d+>|<Audio \d+>|<Video \d+>)",
                         sec.get("subject_definitions", ""), re.M)
    retention = sec.get("retention_analysis", "")
    for lab in defined:
        if lab not in retention:
            out.append(("FAIL", f"{lab} is defined but has no retention line"))
    if re.search(r"\(S\d+\)", retention):
        out.append(("FAIL", "(Sx) speaker id in retention_analysis (ref-en.txt:278)"))

    dd = sec.get(main_field, "")
    for lab in defined:
        if lab.startswith("<Subject") and lab not in dd:
            out.append(("WARN", f"{lab} never cited in {main_field} "
                                f"(ref-en.txt:231)"))

    for m in re.finditer(r"<d>(.*?)</d>", prompt, re.S):
        if not re.match(r"\s*\[[A-Z][a-z]+\]", m.group(1)):
            out.append(("FAIL", "a <d> block has no [Language] tag"))
    for s, body in sec.items():
        if s != main_field and "<d>" in body:
            out.append(("FAIL", f"<d> appears in {s}; it belongs only in "
                                f"{main_field}"))
    out.extend(marker_rules(prompt, dd))
    out.extend(embedding_notes(prompt))

    shots = re.findall(r"\[Shot (\d+)\]([^\n]*)", dd)
    if shots and shots[0][0] == "1" and re.match(r"\s*At \d", shots[0][1]):
        out.append(("FAIL", "[Shot 1] carries a timestamp; it must not"))
    stamps = [int(a) * 60 + float(b)
              for a, b in re.findall(r"At (\d+):(\d+\.\d+)", dd)]
    if stamps != sorted(stamps):
        out.append(("FAIL", "cut timestamps are not strictly increasing"))
    length = node["inputs"].get("length")
    if isinstance(length, int) and stamps:
        dur = h3_rules.duration_of(h3_rules.snap_length(length))
        late = [s for s in stamps if s >= dur]
        if late:
            out.append(("FAIL", f"cut at {late[0]:.3f}s is past the clip's "
                                f"{dur:.3f}s"))

    for line in retention.splitlines():
        mk = re.findall(r"\b(" + "|".join(VISUAL_MARKERS | AUDIO_MARKERS) + r")\b", line)
        is_audio = line.startswith("<Audio")
        for k in mk:
            if is_audio and k not in AUDIO_MARKERS:
                out.append(("FAIL", f"visual marker '{k}' on an audio label"))
            if not is_audio and k not in VISUAL_MARKERS:
                out.append(("FAIL", f"audio marker '{k}' on a visual label"))

    # The 350-500 budget is ref-en.txt:242 and has no counterpart in base-en,
    # so it is NOT applied to a base-format prompt. Applying it there would be
    # inventing a rule -- the mirror image of inventing an exemption, and the
    # same defect.
    if dd and guide == "ref":
        n = len(dd.split())
        editing = "video editing" in sec.get("summary", "")
        if not editing and not (350 <= n <= 500):
            out.append(("WARN", f"detailed_description is {n} words; the guide "
                                f"asks 350-500 for generation tasks "
                                f"(ref-en.txt:242)"))
            # A single frame deviates from that deliberately, and the warning
            # STAYS -- annotated, not suppressed. The distinction matters and
            # is worth keeping straight: the audio sections are exempted above
            # because they are structurally impossible (no decoder, nothing to
            # describe), whereas a word budget is perfectly possible on a still
            # and has simply been judged wrong for it. The guide exempts
            # editing tasks and says nothing about stills, so silencing this
            # would be inventing an exemption -- the exact move this repo
            # criticised the third-party prompt pack for. Annotating keeps the
            # signal honest without crying wolf.
            if _single_frame(node, graph):
                out.append(("note", "single frame: the image path deviates "
                                    "here on purpose (a still has no shot "
                                    "timing, camera-over-time or chronology "
                                    "to describe). See docs/h3_image_editing.md "
                                    "-- deliberate, not an oversight"))

    # base-en's Part One. The exact instruction is parsed from the guide, then
    # its N and S.SS placeholders are resolved from this graph. Presence alone
    # was not a control: an FL2VA graph carrying I2VA's sentence still named a
    # Picture and passed. That wrong sentence existed in `scene_prompt()` on
    # 2026-08-23, staged for the first caller that would have used it.
    if guide == "base" and main_field in sec:
        preamble = prompt[:prompt.index(main_field + ":")].strip()
        kf = [k for k in KEYFRAME_SOCKETS if ins.get(k) is not None]
        if kf and not preamble:
            out.append(("FAIL", f"{'/'.join(kf)} wired but the prompt has no "
                                f"alignment instruction before "
                                f"{main_field} (base-en.txt:19-29)"))
        elif kf and "Picture" not in preamble:
            out.append(("FAIL", "the opening instruction names no Picture "
                                "(base-en.txt:19-29)"))
        elif not kf and preamble:
            out.append(("FAIL", "T2VA begins directly with the core fields, "
                                "but this prompt opens with "
                                f"{preamble.splitlines()[0][:60]!r} "
                                "(base-en.txt:14)"))
        else:
            expected_alignment, mode = _expected_base_alignment(
                node, graph, shots)
            if expected_alignment is None:
                out.append(("FAIL", mode))
            elif preamble != expected_alignment:
                out.append(("FAIL", f"{mode} alignment instruction differs "
                                    "from the release guide after resolving "
                                    "Shot N and S.SS"))
    return out


def price(node: dict, graph: dict) -> list[str]:
    ins = node["inputs"]
    media_ins, image_policies, typed_references = _reference_media(ins, graph)
    w, h = ins.get("width"), ins.get("height")
    length = ins.get("length")
    lines = []
    # An input that is LINKED carries a list, not a literal, and the link is
    # what executes. Follow it to the source node rather than guessing which
    # node feeds it: assuming MiniMaxH3Resolution AND assuming its `wide` shape
    # made every 1024x768 video-reference arm silently unpriceable -- the most
    # OOM-prone graphs in the repo, reporting "cannot price" instead of a
    # number. Read the shape widget to pick the resolution key.
    def follow(val):
        if isinstance(val, int):
            return val, None
        if isinstance(val, list) and val and str(val[0]) in graph:
            return None, graph[str(val[0])]
        return None, None

    srcs = {}
    for field in ("width", "height", "length"):
        lit, src = follow(ins.get(field))
        if lit is None and src is not None:
            srcs[field] = src
    if srcs:
        src = next(iter(srcs.values()))
        si = src.get("inputs", {})
        shape = si.get("shape")
        res = si.get(f"shape.{shape}_resolution") if shape else None
        m = re.match(r"(\d+)x(\d+)", (res or "").strip())
        if m:
            w, h = int(m.group(1)), int(m.group(2))
        if not isinstance(length, int):
            length = si.get("length", length)
        lines.append(f"  (canvas/length followed to "
                     f"{src.get('class_type')}, shape={shape})")
    if not all(isinstance(v, int) for v in (w, h, length)):
        return ["  cannot price: canvas or length not resolvable statically"]

    snapped = h3_rules.snap_length(length)
    # frames -> latent frames. Reproduced from `video_latent_t` at
    # `comfy_extras/nodes_minimax_h3.py:40`, cited because the intuitive guess
    # is wrong in the dangerous direction: assuming 4x temporal compression
    # gives 91 latent steps at 362 frames against the real 107, an 18%
    # under-count that makes an OOM-prone arm look affordable.
    #
    # **The `<= 1` branch is OUR runtime, not core's.** Core returns 2 for any
    # frame_count <= 5, because core clamps to a 5-frame floor and never sees
    # 1. `single_frame.py` lifts that floor, and its own retirement condition
    # asserts `video_latent_t(1) == 1` (`single_frame.py:247`). Every graph in
    # `workflows/image/` is length=1, so copying core's function verbatim into
    # a tool would price all eight of them at double.
    latent_t = 1 if snapped <= 1 else (snapped - 5) // 17 * 5 + 2
    per_frame = rows(w, h)
    video = latent_t * per_frame
    lines.append(f"  video     {video:>8,}  ({latent_t} latent frames x "
                 f"{per_frame:,}/frame at {w}x{h}, {snapped} frames)")

    ref_total = 0
    for key, link in media_ins.items():
        if not key.startswith("ref_images.") or not isinstance(link, list):
            continue
        size_mode = image_policies.get(
            key, ins.get("ref_image_size", "match"))
        src = graph.get(link[0], {})
        # No fit node means core sizes this reference on its own, and core
        # clamps with min(1.0, ...) in BOTH modes
        # (comfy_extras/nodes_minimax_h3.py:297-301), so it never enlarges.
        # Defaulting upscale True here priced a hand-built graph -- the only
        # kind that reaches this branch, since every shipped API graph feeds
        # its references through the fit node -- as if a small reference were
        # raised to 2048, which over-counts its rows by the square of the
        # scale it never gets. Found 2026-08-21 against the core source.
        upscale, short_edge = False, 2048
        fitted = src.get("class_type") == FIT_NODE
        if fitted:
            upscale = src["inputs"].get("allow_upscale", True)
            short_edge = src["inputs"].get("short_edge", 2048)
            inner = src["inputs"].get("image")
            src = graph.get(inner[0], {}) if isinstance(inner, list) else {}
        fname = src.get("inputs", {}).get("image")
        if not fname:
            lines.append(f"  {key}: source not statically resolvable")
            continue
        p = resolve_image(fname)
        if p is None:
            lines.append(f"  {key}: {fname} NOT FOUND in the input dirs")
            continue
        wh = image_size(p)
        if wh is None:
            lines.append(f"  {key}: {fname} unreadable")
            continue
        iw, ih = wh
        full = short_edge / min(iw, ih)
        scale = full if upscale else min(1.0, full)
        if size_mode == "match":
            scale = min(scale, (w * h / (iw * ih)) ** 0.5)
        tw, th = fit(iw, ih, scale)
        r = rows(tw, th)
        ref_total += r
        bound_notes = _vision_bound_warnings(key, tw, th)
        note = "  (fit was a no-op)" if fitted and scale == 1.0 else ""
        if not fitted:
            note = "  (no fit node: core clamps, never upscales)"
        lines.append(f"  ref image {r:>8,}  {fname} {iw}x{ih} -> {tw}x{th}"
                     f"{note}")
        lines.extend(bound_notes)
    if ref_total:
        lines.append(f"  refs      {ref_total:>8,}  total DiT reference rows")

    # Say what is NOT counted, loudly. A reference video is the most expensive
    # input in the model -- 52,020 rows for one 960x544 clip at 345 frames,
    # measured -- and it arrives as an IMAGE batch whose frame count is not
    # knowable from the graph. Omitting it silently would make a number that
    # reads as clearance, which is the failure mode this whole file exists for.
    unpriced = []
    for kind_, prefix in (("video", "ref_videos."),
                          ("video soundtrack", "ref_video_audios."),
                          ("audio", "ref_audios.")):
        n = sum(1 for k, v in media_ins.items()
                if k.startswith(prefix) and v is not None)
        if n:
            unpriced.append(f"{n} {kind_}")
    if unpriced:
        lines.append(f"  NOT COUNTED: {', '.join(unpriced)} reference(s). A "
                     f"video reference alone measured 52,020 rows at 960x544 / "
                     f"345f, so the total below is a FLOOR, not a budget.")
    video_policy = (ins.get("video_policy", "comfy") if typed_references
                    else "native-comfy")
    lines.extend(reference_video_report(
        media_ins, graph, length, video_policy=video_policy))
    lines.extend(audio_reference_notes(
        media_ins, graph, length, typed_boundary=typed_references))

    prompt = ins.get("prompt", "")
    tt, kind = text_tokens(prompt)
    twin = ref_total + 100 if ref_total else 0
    lines.append(f"  text      {tt:>8,}  prompt tokens ({kind})"
                 + (f" + ~{twin:,} vision blocks" if twin else ""))

    # A resolved `embedding:` reference is expanded into the sequence one row
    # per token by `comfy/sd1_clip.py` (`emb.view(1, -1, D)`, then
    # `index += emb_shape - 1`), so it is real width, not a single token
    # standing in for a file.
    emb_rows, emb_unresolved = embedding_tokens(prompt)
    if emb_rows:
        lines.append(f"  embed     {emb_rows:>8,}  rows from resolved "
                     f"`embedding:` reference(s)")
    if emb_unresolved:
        lines.append(f"  NOT COUNTED: {emb_unresolved} `embedding:` "
                     f"reference(s) did not resolve. ComfyUI drops them and "
                     f"renders, so the total below is the price of a prompt "
                     f"missing them -- see the WARN above.")

    # Target audio, omitted entirely until 2026-08-16 and worth 1,206 rows at
    # 362 frames. `PackedLayout` appends it unconditionally between the
    # references and the video -- "target audio then target video, always the
    # last two segments" (`comfy/ldm/minimax/model.py:390-391`) -- so every
    # graph carries it whether or not a soundtrack is wired.
    #
    # Measured against the real layout via `bench/count_packed_rows.py`: this
    # file reported 97,394 for h3_probe_capture_ref3 where the sequence is
    # 98,524. The 1,130 gap is this segment (-1,206) net of the vision-block
    # estimate running 76 high. **The omission ran in the under-pricing
    # direction**, which is the one that matters for a tool whose output is
    # read as headroom before an OOM-prone render.
    # `snapped` is the resolved frame count, the same one the video line
    # prints. Deriving audio from it rather than from the raw `length` input
    # matters: `length` is often a link, and the snap to the 17k+5 grid is what
    # the model actually runs.
    # Imported when ComfyUI is reachable, and NOT restated when it is not.
    # This file's whole premise is that it runs with no CUDA, no model and no
    # server, so ComfyUI's root is not on its path by default -- the import is
    # attempted with the root added, and a failure downgrades the total rather
    # than inventing a rule. `latent_t` above is restated inline and carries a
    # comment defending it; a second inlined rule is a second thing to drift.
    audio_rows = 0
    try:
        # insert(0), not append: `comfy_extras.nodes_minimax_h3` does a bare
        # `import nodes`, and this repo's own `nodes.py` is already on the path
        # at position 0 (line 73). Appending leaves ours winning and the import
        # dies on a relative import -- which is exactly what happened on the
        # first attempt here. Safe to put ComfyUI first: every module this file
        # imports from the repo is already bound by now.
        audio_rows = _core_minimax_cpu().temporal_shape(snapped)[2] * 2
    except Exception:
        lines.append("  NOT COUNTED: target audio rows -- ComfyUI is not "
                     "importable from here, so `temporal_shape` could not be "
                     "read. That segment is ~1,206 rows at 362 frames and is "
                     "always present, so the total below is short by about "
                     "that much.")
    if audio_rows:
        lines.append(f"  audio     {audio_rows:>8,}  target audio rows "
                     f"(always present, soundtrack or not)")

    total = video + ref_total + tt + twin + audio_rows + emb_rows
    lines.append(f"  TOTAL    ~{total:>8,}  packed sequence")
    lines.append("")
    lines.append("  recorded peaks on this box, for judgement not prediction:")
    lines.append("    78,019 tok (2 img refs + 1 video ref, 124f)  ->  21,938 MiB")
    lines.append("   ~124,000 tok (2 img refs upscaled, 362f)      ->  17,840 MiB")
    lines.append("   182,092 tok (imgs at max + video ref, 345f)   ->  OOM")
    return lines


# --- attention chain ------------------------------------------------------
#
# This is the ONE section that runs on a UI-format graph too, and the reason is
# the defect it was written for. On 2026-08-18 a hand-built r2v graph carried an
# ACTIVE `SolAttnMiniMax` whose MODEL output went nowhere: `SageChainAssert`
# took its model straight from the sage node, so ComfyUI -- which seeds
# execution from output nodes and walks backwards -- never ran the Sol node at
# all. The render was dense and looked entirely normal. Measured against the
# same graph with the chain closed, the orphan cost 1.54x on the sampler.
#
# Nothing could have caught it. Every graph-walking check goes through
# `graph_paths()`, which covers `workflows/` and `workflows/image/`; the graph
# lived outside both. And no *node* can see it either: an orphaned node is
# never executed, so there is no runtime moment at which to complain. It is a
# graph-topology defect and a static reader is the only thing that can see it.
#
# The discriminator is `mode`, not presence. Disabling Sol deliberately is done
# by bypassing it (`mode=4`), which is what `build_workflows.py` emits for the
# capture probe and what the UI writes when you press Ctrl-B. An ACTIVE node
# wired to nothing is not a decision anyone makes on purpose, so that is the
# only state reported as a defect. Muted and bypassed are reported as-is.
ATTN_NODES = ("MiniMaxH3SageAttention", "SolAttnMiniMax")
_OUTPUT_TYPES = {"VHS_VideoCombine", "SaveImage", "PreviewImage", "SaveAudio",
                 "SaveAnimatedWEBP", "SaveWEBM", "SaveVideo", "PreviewAny"}


def _reachable_from_outputs(graph):
    """Node ids that feed an output node, for either graph format.

    Reachability, not presence. `docs/evidence.md` records a capture whose
    provenance field reported Sol-free by asking whether the node existed; the
    node existed and was orphaned, so the field read the situation backwards.
    """
    if isinstance(graph.get("nodes"), list):
        nodes = {n["id"]: n for n in graph["nodes"]}
        edges = {}
        for link in graph.get("links", []):
            if isinstance(link, list) and len(link) >= 5:
                edges.setdefault(link[3], set()).add(link[1])
        seeds = [i for i, n in nodes.items() if n.get("type") in _OUTPUT_TYPES]
    else:
        nodes = {i: n for i, n in graph.items() if isinstance(n, dict)}
        edges = {}
        for i, n in nodes.items():
            for val in (n.get("inputs") or {}).values():
                if isinstance(val, list) and len(val) == 2:
                    edges.setdefault(i, set()).add(str(val[0]))
        seeds = [i for i, n in nodes.items()
                 if n.get("class_type") in _OUTPUT_TYPES]
    seen, stack = set(), list(seeds)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(edges.get(cur, ()))
    return seen


def attention_chain(graph):
    """`[(node_type, state, detail)]` for each attention node in the graph.

    States: `live`, `orphaned`, `bypassed`, `muted`, `absent`.
    """
    live_ids = _reachable_from_outputs(graph)
    is_ui = isinstance(graph.get("nodes"), list)
    if is_ui:
        found = {}
        for n in graph["nodes"]:
            found.setdefault(n.get("type"), []).append(n)
    else:
        found = {}
        for i, n in graph.items():
            if isinstance(n, dict):
                found.setdefault(n.get("class_type"), []).append((i, n))

    out = []
    for want in ATTN_NODES:
        got = found.get(want, [])
        if not got:
            out.append((want, "absent", ""))
            continue
        for item in got:
            if is_ui:
                nid, mode = item["id"], item.get("mode", 0)
                if mode == 4:
                    out.append((want, "bypassed", f"node {nid}, mode=4"))
                elif mode == 2:
                    out.append((want, "muted", f"node {nid}, mode=2"))
                elif nid in live_ids:
                    out.append((want, "live", f"node {nid}"))
                else:
                    out.append((want, "orphaned", f"node {nid}"))
            else:
                nid, _node = item
                out.append((want, "live" if nid in live_ids else "orphaned",
                            f"node {nid}"))
    return out


def report_attention_chain(graph):
    lines = ["  attention chain:"]
    defect = False
    for name, state, detail in attention_chain(graph):
        suffix = f"  ({detail})" if detail else ""
        if state == "orphaned":
            defect = True
            lines.append(
                f"    DEFECT  {name} is ACTIVE but nothing consumes its "
                f"MODEL output{suffix}. ComfyUI walks backwards from the "
                f"output nodes, so this node never executes and its patch "
                f"never reaches the sampler. To disable it on purpose, "
                f"bypass it (Ctrl-B) instead -- that is what the shipped "
                f"graphs do.")
        else:
            lines.append(f"    {state:<9}{name}{suffix}")
    if not defect:
        pass
    return lines



# ComfyUI's smart-resize bounds, read from source rather than restated, so this
# goes stale loudly if the shared helper's defaults ever move. They are the
# signature defaults of `process_qwen2vl_images` and of `process_video_block`;
# nothing in the H3 path overrides them. `docs/h3_references.md` owns the
# comparison against what the release declares.
_COMFY_BOUND_RE = re.compile(
    r"min_pixels\s*[:=]\s*int\s*=\s*(\d+)|min_pixels\s*=\s*(\d+)")


def _comfy_image_bounds():
    """(min_pixels, max_pixels) as ComfyUI's shared helper defaults them.

    Returns None when the source cannot be read. A caller that treats that as
    "no problem found" would be reporting silence as clearance, which is the
    failure this whole file exists to avoid, so the caller says so instead.
    """
    for rel in ("comfy/text_encoders/qwen_vl.py",):
        # Same root the prompt-guide reader below resolves, and computed rather
        # than written down: this file is committed and a machine's layout is
        # not a property of the project.
        path = Path.home() / "ComfyUI" / rel
        if not path.exists():
            continue
        text = path.read_text()
        lo = re.search(r"min_pixels:\s*int\s*=\s*(\d+)", text)
        hi = re.search(r"max_pixels:\s*int\s*=\s*(\d+)", text)
        if lo and hi:
            return int(lo.group(1)), int(hi.group(1))
    return None


def _vision_bound_warnings(key, tw, th):
    """Warn when a reference will hit a bound ComfyUI and the release disagree on.

    Neither bound binds on any graph this repo ships -- the fit node puts every
    reference at a 2048 short edge, which is above the release's floor and below
    ComfyUI's ceiling until roughly 3:1. This exists to say WHEN that stops
    being true, because the fix for it is a monkeypatch nobody should write on
    speculation.
    """
    out = []
    pixels = tw * th
    rel_lo, rel_hi = vendor_config.image_pixel_bounds()
    comfy = _comfy_image_bounds()
    if comfy is None:
        return [f"  {key}: could not read ComfyUI's pixel bounds; the "
                f"release/ComfyUI comparison was NOT made for this reference"]
    c_lo, c_hi = comfy
    if pixels > c_hi >= 0 and c_hi < rel_hi:
        out.append(
            f"  {key}: WARN {tw}x{th} is {pixels:,} px, above ComfyUI's "
            f"{c_hi:,} ceiling but inside the release's {rel_hi:,}. The "
            f"conditioner will shrink it; the released pipeline would not.")
    if pixels < rel_lo and pixels >= c_lo:
        out.append(
            f"  {key}: WARN {tw}x{th} is {pixels:,} px, under the release's "
            f"{rel_lo:,} floor but above ComfyUI's {c_lo:,}. The released "
            f"pipeline would enlarge it to the floor; the conditioner will not.")
    return out

def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__.strip().splitlines()[0])
        print("\nusage: python bench/preflight_graph.py <graph.json> [...]")
        return 0
    for path in paths:
        print(f"\n=== {path}")
        try:
            graph = json.loads(path.read_text())
        except Exception as exc:
            print(f"  cannot read: {exc}")
            return 1
        for line in report_attention_chain(graph):
            print(line)
        print("")
        if isinstance(graph.get("nodes"), list):
            print("  UI-format graph; the PRICING half reads the API form "
                  "(links are resolved there). Skipped. The attention chain "
                  "above was read from the UI form and is complete.")
            continue
        refs = {nid: n for nid, n in graph.items()
                if n.get("class_type") in PROMPT_NODES}
        if not refs:
            print(f"  no {' or '.join(PROMPT_NODES)}; nothing to grade")
            continue
        for node in refs.values():
            findings = grade(node, graph, path.stem)
            if not findings:
                print("  prompt: all mechanical rules pass")
            for level, msg in findings:
                print(f"  {level:<4}  {msg}")
            print("")
            for line in price(node, graph):
                print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
