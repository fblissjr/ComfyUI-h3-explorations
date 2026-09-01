#!/usr/bin/env python3
"""Grade a graph's prompt and price its sequence, BEFORE you press Queue.

    python bench/preflight_graph.py internal/refs/test.json
    python bench/preflight_graph.py workflows/*_api.json

**One glob per graph directory, and that is not a style choice.**
`workflows/*_api.json` is non-recursive, so it prices exactly the graphs sitting
directly in `workflows/` and silently skips any subdirectory. That currently
misses nothing -- `h3_config.GRAPH_DIRS` is `("",)` since the single-frame lane
was parked on 2026-08-27 -- and it missed the image graphs for the eleven days
`workflows/image/` existed before that: a one-directory invocation priced a
subset with no error, no warning, just a smaller number nobody had a prior for.
Check `GRAPH_DIRS` before trusting a single glob, or pass the paths explicitly.

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
PROMPT_NODES = ("MiniMaxH3ReferenceToVideo", "MiniMaxH3ImageToVideo",
                "MiniMaxH3Conditioning", "MiniMaxH3ReferenceConditioning")
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
# `internal/PROMPTING.md` (retired 2026-09-01, migrated into
# `docs/prompting.md`) gained a "must" stating the nesting rule, and
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

# One shot header and the text up to the NEXT header, which is what makes this
# safe on a base-mode prompt where every shot shares a line. A greedy
# `[^\n]*` swallows later headers and reports the final shot as 1, and the
# consequence is not a missed diagnostic: `_expected_base_alignment` resolves
# `Shot N` from it, so a wrong parse tells an author to write a guide-violating
# Part One line. Named here because `grade_prompt_text.py` prints the same
# expectation and had drifted to the pre-fix copy while its comment claimed
# they matched.
SHOT_HEADER_RE = r"\[Shot (\d+)\]((?:(?!\[Shot \d+\])[^\n])*)"


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
    # A house-format warning, not tokenizer compatibility logic. The
    # `hard_cut` audit stressor legitimately triggers it, so this reports and
    # never refuses the graph.
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
    """Socket-shaped media view plus per-image sizing for either surface.

    The typed chain remains the source of order and ownership; this adapter is
    only for older pricing/reporting code whose natural unit is a media link.
    Chain traversal comes from ``reference_order`` so a malformed chain cannot
    be priced as a shorter, apparently valid one.

    The per-image entry is the WHOLE stage-one decision -- policy, upscale and
    short edge together -- because that is where all three now live. This file
    read `size_policy` and never read `allow_upscale`, which was not an
    oversight: `allow_upscale` lived on a node the chain model could not see,
    so it had to be recovered by walking upstream and guessing the shape. It
    is a field on the append node now.
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
            # A widget converted to an input socket carries `[node_id, slot]`,
            # not a value. `int([...])` raised TypeError and killed the whole
            # report, and `bool([...])` is True for any non-empty list -- so a
            # converted `allow_upscale` was silently priced as upscaling, a 4x
            # error with no marker. Widget-to-input conversion is an ordinary
            # frontend action, so both are graphs a user can build. Unresolved
            # values become None and the caller says so.
            absent = []

            def _value(name, default, group=None):
                """Read a widget spelled flat OR as a DynamicCombo member.

                `size_policy` became a DynamicCombo on 2026-08-27, so its
                members are `size_policy.short_edge` and
                `size_policy.allow_upscale` in API form. This read the flat
                names only, so `allow_upscale` fell to its default False and
                every shipped reference graph was priced as though it did not
                upscale -- 2,368 DiT rows against the 9,408 a render packs,
                on the tool whose job is to say whether a graph will fit.

                Missing under BOTH spellings is RECORDED rather than
                defaulted. A silent default is what hid the rename, and this
                file already treats an unreadable value as "not priced"
                rather than as a guess.
                """
                # `dit_short_edge` was `short_edge` until 2026-08-28. An old
                # hand-built graph still carries the old spelling, and pricing
                # it wrong is worse than pricing it -- so the previous name is
                # tried too. `absent` still reports the CURRENT name, because
                # that is what a reader has to go and write.
                aliases = {"dit_short_edge": ("short_edge",)}
                names = (name,) + aliases.get(name, ())
                cands = []
                for nm in names:
                    cands += [f"{group}.{nm}", nm] if group else [nm]
                for candidate in cands:
                    if candidate in append_inputs:
                        raw = append_inputs[candidate]
                        return None if isinstance(raw, list) else raw
                absent.append(name)
                return default

            size_policy = _value("size_policy", "match")
            allow_upscale = _value("allow_upscale", False, group="size_policy")
            # Renamed on the node 2026-08-28; the internal dict key below
            # stays `short_edge` because it is preflight's own, and the
            # retired fit node further down still has an input of that
            # name that must NOT follow this rename.
            short_edge = _value("dit_short_edge", 2048, group="size_policy")
            # `qwen_view` is a DynamicCombo since 2026-08-31, replacing a flat
            # `qwen_short_edge` Int whose 0 meant "shared view". Same rename
            # shape as `size_policy` above, and this reader refused to price
            # rather than default when it hit it -- which is the behaviour the
            # docstring above argues for, working.
            #
            # `shared` carries no size member at all, so reading it as absent
            # would report a defect on a correct graph. It maps to 0, which is
            # still preflight's internal spelling for one shared view.
            qwen_view = _value("qwen_view", None)
            if qwen_view == "shared":
                if "qwen_view" in absent:
                    absent.remove("qwen_view")
                qwen_short_edge = 0
            elif qwen_view is not None:
                # A `separate` view always carries a size, so the schema
                # default is the right fallback rather than None.
                # 512 is `h3_config.REF_QWEN_SHORT_EDGE` and the node's schema
                # default, written as a literal because this reader is
                # deliberately standalone -- same as the 2048 above, which is
                # `REF_IMAGE_SHORT_EDGE`. Inherited, not measured: it rests on
                # one render at one seed.
                qwen_short_edge = _value("qwen_short_edge", 512,
                                         group="qwen_view")
            else:
                # Neither spelling: an older graph on the flat Int, or a
                # hand-built one. 0 is what the retired node did with the key
                # missing, so it is a reading of that graph rather than a guess.
                #
                # **The default must not be None here.** `_value` returns None
                # for a value WIRED to another node, and `linked` below is
                # computed from exactly that -- so a None default reports an
                # absent input as linked, which is a different defect entirely.
                qwen_short_edge = _value("qwen_short_edge", 0)
            image_policies[key] = {
                "size_policy": size_policy,
                "allow_upscale": (None if allow_upscale is None
                                  else bool(allow_upscale)),
                "short_edge": None if short_edge is None else int(short_edge),
                "qwen_short_edge": (None if qwen_short_edge is None
                                    else int(qwen_short_edge)),
                "linked": [n for n, v in (("size_policy", size_policy),
                                          ("allow_upscale", allow_upscale),
                                          ("short_edge", short_edge),
                                          ("qwen_short_edge", qwen_short_edge))
                           if v is None],
                "absent": absent,
            }
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
_EMBEDDING_DIRS: list[Path] | None = None
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

    Resolved once and cached. This is called per `embedding:` reference, and an
    earlier version inserted into `sys.path` on every call -- five references
    left five duplicate entries, unbounded across a sweep of every graph.
    """
    global _EMBEDDING_DIRS
    if _EMBEDDING_DIRS is not None:
        return _EMBEDDING_DIRS
    dirs = []
    try:
        comfy_root = str(Path.home() / "ComfyUI")
        if comfy_root not in sys.path:
            sys.path.insert(0, comfy_root)
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
    _EMBEDDING_DIRS = out
    return out


def resolve_embedding(name: str) -> tuple[str, str, int]:
    """(state, detail, token rows) for one `embedding:` reference. Never raises.

    The row count is RETURNED, not recovered from `detail` by regex. It was, and
    that coupled pricing to the wording of a human-readable sentence: reword
    "7 token(s)" and the sequence total silently drops those rows while still
    reporting a number, which is the shape of under-report this file exists to
    stop.

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
            f"{', '.join(str(d) for d in _embedding_dirs())}"), 0
    if hit.suffix != ".safetensors":
        # Only the safetensors header is cheap to read without torch.
        return "unreadable", f"{hit.name} is not safetensors; header not read", 0
    try:
        from safetensors import safe_open
        with safe_open(hit, framework="pt") as f:
            keys = list(f.keys())
            key = next((k for k in (H3_EMBEDDING_KEY, H3_EMBEDDING_ALT_KEY)
                        if k in keys), None)
            if key is None:
                return "unreadable", (
                    f"{hit.name} holds {keys}, neither "
                    f"`{H3_EMBEDDING_KEY}` nor `{H3_EMBEDDING_ALT_KEY}`"), 0
            shape = list(f.get_slice(key).get_shape())
    except Exception as exc:
        return "unreadable", f"{hit.name}: {type(exc).__name__}: {exc}", 0
    if len(shape) != 2 or shape[-1] != H3_HIDDEN_SIZE:
        return "unreadable", (
            f"{hit.name} key `{key}` is {shape}, not "
            f"[tokens, {H3_HIDDEN_SIZE}]"), 0
    return "ok", f"{hit.name} `{key}` {shape[0]} token(s)", int(shape[0])


def embedding_notes(prompt: str) -> list[tuple[str, str]]:
    """Findings for every `embedding:` reference in a prompt."""
    out = []
    for name in _EMBEDDING_REF.findall(prompt or ""):
        state, detail, _rows = resolve_embedding(name)
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
        state, _detail, got = resolve_embedding(name)
        if state != "ok":
            unresolved += 1
            continue
        rows += got
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
        cfg = json.loads((_REPO / "vendor_config"
                          / "video_preprocessor_config.json").read_text())
        return _qwen_grid_from(n_raw, w, h, cfg)
    except Exception:
        return None


def _qwen_grid_from(n_raw: int, w: int, h: int, cfg: dict):
    """One configured video processor's grid, executed. Raises on failure."""
    import torch
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        Qwen3VLVideoProcessor,
    )
    proc = Qwen3VLVideoProcessor(
        **{k: v for k, v in cfg.items()
           if k not in ("processor_class", "video_processor_type")})
    out = proc(videos=[torch.zeros(n_raw, 3, h, w, dtype=torch.uint8)],
               do_sample_frames=False, input_data_format="channels_first",
               return_tensors="pt")
    g = out["video_grid_thw"][0].tolist()
    merge = int(proc.merge_size) ** 2
    return g, g[1] * g[2] // merge


def _contract_qwen_grid(n_raw: int, w: int, h: int, contract: dict):
    """The loaded encoder's Qwen video grid, from its stamped contract.

    Same executed processor as `_release_qwen_grid`, configured from the
    contract rather than the release file, so a v2 directory carrying the
    release bounds and the v1 snapshot carrying its own both price at what
    they declare. Returns None when it cannot be computed; the caller says so.
    """
    try:
        lo, hi = contract["video_bounds"]
        cfg = {"size": {"shortest_edge": lo, "longest_edge": hi},
               **contract["video_geometry"]}
        return _qwen_grid_from(n_raw, w, h, cfg)
    except Exception:
        return None


def _encoder_contract_for(ins: dict, graph: dict):
    """Resolve the conditioner's `encoder` policy from the node feeding `clip`.

    Static counterpart of `reference_geometry.encoder_contract_from_clip`: the
    preflight sees a graph, not a loaded CLIP, so it walks the `clip` link to
    the loader node and asks the adapter what that artifact declares. Core's
    `CLIPLoader` declares nothing, and so does a name the adapter does not
    know; both come back `None` with the reason, and the caller prices
    `encoder` as the native path it will actually run.
    """
    link = ins.get("clip")
    if not (isinstance(link, list) and link and str(link[0]) in graph):
        return None, "clip input is not linked, so no encoder contract"
    src = graph[str(link[0])]
    kind = src.get("class_type")
    if kind == "MiniMaxH3EncoderLoader":
        # The guarded loader stamps what CORE's own preprocessing will do, so
        # the static answer is that same derivation rather than an artifact's
        # declaration -- and it must not stay "native, unresolved" here while
        # the runtime has a contract, or the preflight prices a different
        # encoder than the render uses.
        _core_minimax_cpu()
        import h3_encoder_loader
        contract = h3_encoder_loader.native_encoder_contract()
        return contract, f"encoder contract from ComfyUI's own H3 path ({contract['source']})"
    if kind != "MiniMaxH3AWQEncoderLoader":
        return None, f"{kind} declares no processor contract; encoder = native"
    name = src.get("inputs", {}).get("encoder_name")
    if not isinstance(name, str):
        return None, ("MiniMaxH3AWQEncoderLoader.encoder_name is linked, not a "
                      "literal; encoder contract unresolved")
    _core_minimax_cpu()
    import h3_awq_encoder
    contract = h3_awq_encoder.encoder_contract_from_artifact(name)
    if contract is None:
        return None, f"{name!r} is not an artifact this adapter knows; encoder = native"
    return contract, f"encoder contract from {name!r} ({contract['source']})"


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
    ins: dict, graph: dict, length, video_policy: str = "native-comfy",
    contract: dict | None = None,
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
        enc = (_contract_qwen_grid(raw, *comfy_vae, contract)
               if video_policy == "encoder" and contract else None)

        out.append(f"  {label}: {name}")
        out.append(f"      source                {w}x{h}")
        comfy_active = video_policy in ("comfy", "native-comfy")
        active_kind = ("native core" if video_policy == "native-comfy"
                       else "local typed policy")
        # The encoder policy keeps the no-upscale VAE view and runs the
        # contract's processor on it, so its VAE line is comfy's.
        vae_comfy_active = comfy_active or video_policy == "encoder"
        out.append(f"      VAE-prepared, comfy   {comfy_vae[0]}x{comfy_vae[1]}"
                   f"{'   (gap 6: not upscaled)' if comfy_vae != rel_vae else ''}"
                   f"{'   <- ACTIVE (' + active_kind + ')' if vae_comfy_active else ''}")
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
        if video_policy == "encoder":
            if enc is None:
                out.append(f"      Qwen, encoder         NOT CALCULATED -- "
                           f"{'no encoder contract resolved from the graph' if not contract else 'the contract processor could not be run'}; "
                           f"the encoder policy on this graph runs the native "
                           f"per-pair path above unless a loader declares one")
            else:
                g, per = enc
                out.append(f"      Qwen, encoder         {g[2] * 16}x{g[1] * 16}  "
                           f"{per * g[0]:>7,} rows   <- ACTIVE (local typed "
                           f"policy; contract {(contract or {}).get('source')})")
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


def rows(w: int, h: int) -> int:
    """DiT rows for a w x h pixel area: VAE /16, then the DiT's 2x2 patch.

    Kept as a thin local name for the callers that price a canvas rather than a
    reference; it is `reference_geometry.latent_rows` and must stay so. The
    sibling `fit()` was deleted on 2026-08-24 once the reference path stopped
    using it -- a dead copy of shared arithmetic is exactly how the fourth copy
    comes back.
    """
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


def _resolved_prompt(node: dict, graph: dict) -> str | None:
    """Return an inline prompt or follow a frontend string primitive once.

    ComfyUI's public Ref2VA workflow keeps its prompt in a
    `PrimitiveStringMultiline` node. The API graph therefore hands the
    conditioner a link, not a string. Treating that two-item link as text used
    to crash `split_sections`, so the exact HF workflow could not be
    preflighted even though ComfyUI executes it.
    """
    value = node.get("inputs", {}).get("prompt", "")
    seen = set()
    while isinstance(value, list) and len(value) == 2:
        source_id = str(value[0])
        if source_id in seen or source_id not in graph:
            return None
        seen.add(source_id)
        source = graph[source_id]
        inputs = source.get("inputs", {})
        value = inputs.get("value", inputs.get("text"))
    return value if isinstance(value, str) else None


def text_tokens(prompt: str) -> tuple[int, str]:
    """Token count from ComfyUI's own tokenizer, or an estimate if absent.

    **The special tokens are added, and that is not cosmetic.** The runtime
    tokenizer is `MiniMaxQwenSDTokenizer`, which registers the release's
    `additional_special_tokens` before tokenizing, so each H3 marker is ONE id.
    A plain load of the same directory splits them into pieces instead, and
    this function reported "exact" while being wrong by one token per marker
    part -- 786 against the runtime's 784 on the market ref2va prompt, which
    carries `<d>` and `</d>`.

    Small, and it was still worth fixing: the label said exact, the error is
    silent, and it grows with exactly the prompts this repo is most interested
    in, since a marker scene is mostly markers. Read the list from
    `vendor_config`, never retyped -- it is the release's own declaration.
    """
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(
            Path.home() / "ComfyUI" / "comfy" / "text_encoders" / "qwen25_tokenizer")
        try:
            sys.path.insert(0, str(_REPO))
            import vendor_config
            extra = [t for t in vendor_config.additional_special_tokens()
                     if t not in tk.get_vocab()]
            if extra:
                tk.add_special_tokens({"additional_special_tokens": extra})
        except Exception:
            # Without the release on disk the count is still far better than
            # the word estimate; say so rather than silently claiming exact.
            return len(tk(prompt)["input_ids"]), "exact-, no marker list"
        return len(tk(prompt)["input_ids"]), "exact"
    except Exception:
        return int(len(prompt.split()) * 1.35), "~est"


def _single_frame(node: dict, graph: dict) -> bool:
    """True when this graph renders one frame. Read off the graph, never a name.

    `length` is usually LINKED rather than a literal (the parked image graphs
    carried `['27', 2]`), so a literal-only read misses the graphs that matter.
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
        source = graph[source_id]
        source_inputs = source.get("inputs", {})
        if source.get("class_type") == "ComfyMathExpression":
            # The official H3 frontend workflows convert a duration primitive
            # with this exact public snap expression and link the INT output.
            # Resolve that graph shape without evaluating arbitrary text.
            expression = source_inputs.get("expression", "")
            duration_link = source_inputs.get("values.a")
            if expression != (
                "max(5, round(a * 24)) + "
                "(5 - (max(5, round(a * 24)) % 17)) % 17"
            ) or not (isinstance(duration_link, list) and len(duration_link) == 2):
                return None
            duration_node = graph.get(str(duration_link[0]), {})
            seconds = duration_node.get("inputs", {}).get("value")
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
                return None
            frames = max(5, round(seconds * 24))
            value = frames + (5 - frames % 17) % 17
        else:
            value = source_inputs.get("length")
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
    prompt = _resolved_prompt(node, graph)
    if prompt is None:
        return [("FAIL", "prompt is linked but its string source could not be resolved")]
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

    # STOP EACH SHOT'S BODY AT THE NEXT SHOT HEADER, not at end of line. The
    # guides put the whole description in ONE unbroken paragraph with
    # `[Shot 2]` running inline (base-en's own worked examples do this), so a
    # greedy `[^\n]*` swallowed every later header and `findall` returned a
    # SINGLE pair for a multi-shot prompt. `shots[-1][0]` was then "1" whatever
    # the prompt actually ended on, and `_expected_base_alignment` demanded
    # `from Shot 1` -- telling an author to write a guide-violating line.
    #
    # It bit nothing shipped because every shipped fl2va and l2va prompt is one
    # shot, and the t2va path returns before reading `shots` at all. Found
    # 2026-09-01 writing the first multi-shot keyframe examples.
    shots = re.findall(SHOT_HEADER_RE, dd)
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
                                    "to describe) -- deliberate, not an "
                                    "oversight. That path is PARKED and ships "
                                    "no graphs; see docs/h3_image_editing.md"))

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
    # Stage two is a property of the conditioner, not of any one reference.
    image_policy = ins.get("image_policy", "comfy") if typed_references else "comfy"
    if isinstance(image_policy, list):
        image_policy = "comfy"
    w, h = ins.get("width"), ins.get("height")
    length = ins.get("length")
    lines = []
    # `encoder` means whatever the loaded encoder declares, and the graph
    # says which encoder that is. Resolved once here and used for both the
    # still and the video stage, the way the conditioner does at runtime.
    contract, contract_note = (_encoder_contract_for(ins, graph)
                               if typed_references else (None, None))
    requested_image_policy = image_policy
    if image_policy == "encoder" and contract is None:
        image_policy = "comfy"
    if typed_references and contract_note and (
            requested_image_policy == "encoder"
            or ins.get("video_policy") == "encoder"):
        lines.append(f"  encoder policy: {contract_note}")
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
    # **The `<= 1` branch is NOT core's.** Core returns 2 for any frame_count
    # <= 5, because it clamps to a 5-frame floor and never sees 1. The pack's
    # shim lifted that floor and asserted `video_latent_t(1) == 1` as its own
    # retirement condition; it is parked (`archive/single_frame.py`,
    # 2026-08-27) along with the length=1 graphs it served, so no SHIPPED graph
    # reaches this branch today. It stays because this tool reads whatever path
    # it is handed -- an archived graph, a hand-built one, a patched core --
    # and copying core's function verbatim would price a one-frame graph at
    # double.
    latent_t = 1 if snapped <= 1 else (snapped - 5) // 17 * 5 + 2
    per_frame = rows(w, h)
    video = latent_t * per_frame
    lines.append(f"  video     {video:>8,}  ({latent_t} latent frames x "
                 f"{per_frame:,}/frame at {w}x{h}, {snapped} frames)")

    ref_total = 0
    qwen_total = 0
    qwen_priced = qwen_refs = 0
    for key, link in media_ins.items():
        if not key.startswith("ref_images.") or not isinstance(link, list):
            continue
        policy = image_policies.get(key)
        src = graph.get(link[0], {})
        # Core sizes an unmanaged reference on its own, and it clamps with
        # min(1.0, ...) in BOTH modes (comfy_extras/nodes_minimax_h3.py:297-301),
        # so it never enlarges. Defaulting upscale True here priced a
        # hand-built graph as if a small reference were raised to 2048, which
        # over-counts its rows by the square of the scale it never gets. Found
        # 2026-08-21 against the core source.
        upscale, short_edge = False, 2048
        if policy is None:
            size_mode = ins.get("ref_image_size", "match")
        else:
            if policy["linked"]:
                lines.append(
                    f"  {key}: {', '.join(policy['linked'])} is wired to an "
                    f"input socket, so its value is not in the graph. This "
                    f"reference was NOT priced.")
                continue
            if policy.get("absent"):
                lines.append(
                    f"  {key}: {', '.join(policy['absent'])} is on neither the "
                    f"flat nor the size_policy.* spelling, so this node's "
                    f"schema is not one this reader knows. This reference was "
                    f"NOT priced -- pricing it against a default is how the "
                    f"DynamicCombo rename went unnoticed.")
                continue
            # The typed append owns all three since the fit fold.
            size_mode = policy["size_policy"]
            upscale = policy["allow_upscale"]
            short_edge = policy["short_edge"]
        # A saved graph may still wire the retired fit node upstream. The two
        # COMPOSE -- the fit sizes the source, then the append sizes that
        # result -- so they must be applied in order. Merging them by taking
        # the larger short edge and OR-ing the upscale flags, as this did until
        # 2026-08-24, over-prices by the square of the ratio whenever the
        # append is narrower than the fit: Fit(2048, upscale) -> Append(1024)
        # really yields 1024x1024 and was priced at 2048x2048.
        legacy_fit = None
        if src.get("class_type") == FIT_NODE:
            fit_inputs = src["inputs"]
            legacy_fit = {
                "short_edge": (2048 if isinstance(fit_inputs.get("short_edge"), list)
                               else int(fit_inputs.get("short_edge", 2048))),
                "allow_upscale": (
                    True if isinstance(fit_inputs.get("allow_upscale"), list)
                    else bool(fit_inputs.get("allow_upscale", True))),
            }
            inner = fit_inputs.get("image")
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
        # The shared implementation, not a fourth copy of the arithmetic. This
        # file carried its own until 2026-08-24, which is precisely where a
        # divergence would mislead: preflight exists to price what you are
        # about to render, so a copy that disagrees with the node reports a
        # sequence length nobody will get. Imported lazily, the same way this
        # file already reaches `reference_order` and core.
        _core_minimax_cpu()   # puts ComfyUI on sys.path, CPU-forced
        from reference_geometry import (fit_reference_image, latent_rows,
                                        qwen_image_size)
        stage_w, stage_h = iw, ih
        if legacy_fit is not None:
            stage_w, stage_h = fit_reference_image(
                iw, ih, size_policy="max", short_edge=legacy_fit["short_edge"],
                allow_upscale=legacy_fit["allow_upscale"])
        tw, th = fit_reference_image(
            stage_w, stage_h, size_policy=size_mode, short_edge=short_edge,
            allow_upscale=upscale, canvas_w=w, canvas_h=h)
        # The Qwen view. 0 (or an unmanaged/legacy reference) means the
        # encoder sees the VAE tensor; N means a separate view of the SOURCE
        # at an N short edge for the encoder alone, the VAE keeping stage one.
        qwen_edge = (policy or {}).get("qwen_short_edge") or 0
        qwen_w, qwen_h = (tw, th)
        # Stage two. With no Qwen view of its own, the conditioner applies the
        # selected still policy BEFORE the VAE, so under `encoder` or
        # `release` the geometry the DiT gets is not the role size. Omitting
        # this over-priced an `encoder` graph by more than 10x, on the tool
        # whose job is to price the sequence. With a Qwen view the VAE keeps
        # the role size and stage two shapes the Qwen view only.
        if qwen_edge:
            qwen_w, qwen_h = _qwen_view_size(iw, ih, qwen_edge)
        if image_policy != "comfy":
            try:
                if qwen_edge:
                    qwen_w, qwen_h = qwen_image_size(qwen_w, qwen_h, image_policy, contract)
                else:
                    tw, th = qwen_image_size(tw, th, image_policy, contract)
                    qwen_w, qwen_h = tw, th
            except Exception as exc:
                lines.append(f"  {key}: could not apply image_policy="
                             f"{image_policy!r} ({exc}); priced at the role size")
        scale = tw / iw
        r = latent_rows(tw, th)
        ref_total += r
        bound_notes = (_vision_bound_warnings(key, tw, th)
                       if image_policy == "comfy" and not qwen_edge else [])
        if policy is None:
            note = "  (unmanaged: core clamps, never upscales)"
        elif scale == 1.0:
            note = "  (sizing was a no-op)"
        else:
            note = ""
        if image_policy != "comfy":
            note += f"  (image_policy={image_policy})"
        lines.append(f"  ref image {r:>8,}  {fname} {iw}x{ih} -> {tw}x{th}"
                     f"{note}")
        # The encoder's own processor applies its bounds to whatever it is
        # handed, so the Qwen rows are priced under the loaded encoder's
        # contract when the graph declares one and under Comfy's defaults
        # otherwise -- for every image_policy, `comfy` included.
        qwen_refs += 1
        priced = _qwen_tokens(qwen_w, qwen_h, contract)
        if priced is None:
            lines.append(f"      qwen view {qwen_w}x{qwen_h}: tokens NOT CALCULATED "
                         f"(the stage-two bounds could not be read)")
        else:
            pw, ph, tokens, owner = priced
            qwen_total += tokens
            qwen_priced += 1
            clamp = (f"  <- clamped by the {owner} bounds"
                     if (pw, ph) != (qwen_w, qwen_h) else "")
            view = (f"qwen_short_edge={qwen_edge}" if qwen_edge
                    else "same tensor as the VAE view")
            lines.append(f"      qwen view {qwen_w}x{qwen_h} -> {pw}x{ph}  "
                         f"{tokens:>7,} tokens  ({view}){clamp}")
        lines.extend(bound_notes)
    if ref_total:
        lines.append(f"  refs      {ref_total:>8,}  total DiT reference rows")
    if qwen_total:
        lines.append(f"  qwen      {qwen_total:>8,}  total reference tokens in the "
                     f"text segment, before the prompt")

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
    if isinstance(video_policy, list):
        video_policy = "comfy"
    if video_policy == "encoder" and contract is None:
        # The same substitution the conditioner makes: a CLIP that declares
        # nothing runs the native per-pair path under `encoder`.
        video_policy = "comfy"
    lines.extend(reference_video_report(
        media_ins, graph, length, video_policy=video_policy,
        contract=contract))
    lines.extend(audio_reference_notes(
        media_ins, graph, length, typed_boundary=typed_references))

    prompt = ins.get("prompt", "")
    # The `embedding:name` literal is REPLACED by the embedding's rows, not
    # tokenized alongside them (`comfy/sd1_clip.py` drops the reference and
    # splices the tensor in its place). Counting the raw prompt and then adding
    # the rows charges for both, contradicting the citation below. Strip the
    # literals first, then add the rows they stand for.
    tt, kind = text_tokens(_EMBEDDING_REF.sub("", prompt))
    # The vision blocks in the TEXT segment are the QWEN view, not the DiT
    # reference rows. Those two are equal only while `qwen_short_edge` is 0 --
    # which was every graph ever shipped, because under the v1 encoder's
    # snapshot bounds the knob could not do anything. v2 (2026-08-27) made it
    # live, and this line still read `ref_total`, so the `qwen` line above
    # honoured the knob while the TOTAL ignored it: at a 512 short edge on two
    # upscaled references they differ by 8,816 tokens and the total did not
    # move. A pricing tool that cannot see the one input that reduces the
    # segment is worse than none, because it reads as evidence the knob is inert.
    #
    # Falls back to `ref_total` only when a reference could not be priced at
    # all, and says so rather than quietly reporting a smaller number.
    qwen_incomplete = qwen_refs and qwen_priced < qwen_refs
    twin = (ref_total if qwen_incomplete else qwen_total)
    twin = twin + 100 if twin else 0
    lines.append(f"  text      {tt:>8,}  prompt tokens ({kind})"
                 + (f" + ~{twin:,} vision blocks" if twin else "")
                 + ("  (vision blocks fell back to the DiT row count: a "
                    "reference could not be priced)" if qwen_incomplete else ""))

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
    # These three rows used to sit here under "for judgement not prediction",
    # directly beneath the computed TOTAL, and the layout defeated the label.
    # On 2026-08-28 a 117,770-token arm was read against them as comfortable --
    # below the only OOM row, just under a LARGER row that had used 17.8 GiB --
    # and it OOMed. The rows are also non-monotonic (78k cost MORE than 124k),
    # so there was never a curve to interpolate; three ascending token counts
    # with a trailing OOM merely looked like one.
    lines.append("  peak memory is NOT a function of sequence length, so this")
    lines.append("  section predicts NOTHING about the graph above:")
    lines.append("    78,019 tok (2 img refs + 1 video ref, 124f)  ->  21,938 MiB")
    lines.append("   ~124,000 tok (2 img refs upscaled, 362f)      ->  17,840 MiB")
    lines.append("   182,092 tok (imgs at max + video ref, 345f)   ->  OOM")
    lines.append("    117,770 tok (1 img ref upscaled, 362f)       ->  BOTH:")
    lines.append("      OOMed applying a PDD LoRA onto a model loaded without")
    lines.append("      one, then succeeded on retry once one was resident.")
    lines.append("  Note rows 1 and 2: MORE tokens, LESS memory. What actually")
    lines.append("  moves the peak is resident weights and what ran before --")
    lines.append("  a PDD LoRA adds ~1.0 GiB, and applying it is a transition")
    lines.append("  cost the steady state does not show.")
    lines.append("  bench/results/2026-08-28_pdd_ref2va_memory_marginality.json")
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
# `graph_paths()`, which covers only what `GRAPH_DIRS` names; the graph lived
# outside it. And no *node* can see it either: an orphaned node is
# never executed, so there is no runtime moment at which to complain. It is a
# graph-topology defect and a static reader is the only thing that can see it.
#
# The discriminator is `mode`, not presence. Disabling Sol deliberately is done
# by bypassing it (`mode=4`), which is what `build_workflows.py` emits for the
# capture probe and what the UI writes when you press Ctrl-B. An ACTIVE node
# wired to nothing is not a decision anyone makes on purpose, so that is the
# only state reported as a defect. Muted and bypassed are reported as-is.
# `MiniMaxH3SolAttn` is ours since 2026-08-30 and was ABSENT here until
# 2026-08-31, so the active-but-unwired report could not fire on the node
# every shipped graph now carries. The vendored id stays: saved graphs
# predating the switch still wire it.
ATTN_NODES = ("MiniMaxH3SageAttention", "SolAttnMiniMax", "MiniMaxH3SolAttn")
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


def _qwen_view_size(source_w: int, source_h: int, qwen_short_edge: int):
    """The conditioner's own view: the shared implementation, imported."""
    _core_minimax_cpu()
    from reference_geometry import snap_to_multiple
    scale = qwen_short_edge / min(source_w, source_h)
    return snap_to_multiple(source_w, scale), snap_to_multiple(source_h, scale)


def _qwen_tokens(w: int, h: int, contract):
    """`(w, h, merged_tokens, owner)` after the stage-two bounds, or None.

    Under a stamped contract that is the loaded encoder's declaration; with
    none, Comfy's shared helper defaults, which is what a native CLIP applies.
    Executed through the shared `qwen_image_size` (the processor's own
    `smart_resize`), not modelled. Under the current W4 artifact's snapshot
    this is where a large Qwen view collapses back to about 265 tokens,
    which is the loud caveat on the `qwen_short_edge` knob.
    """
    _core_minimax_cpu()
    from reference_geometry import qwen_image_size
    if contract is not None:
        effective, owner = contract, f"encoder contract ({contract['source']})"
    else:
        bounds = _comfy_image_bounds()
        if bounds is None:
            return None
        sys.path.insert(0, str(_REPO))
        import vendor_config
        effective = {"image_bounds": bounds,
                     "image_geometry": vendor_config.patch_geometry()}
        owner = "native ComfyUI"
    try:
        pw, ph = qwen_image_size(w, h, "encoder", effective)
    except Exception:
        return None
    return pw, ph, (pw // 32) * (ph // 32), owner


def _vision_bound_warnings(key, tw, th):
    """Warn when a reference will hit a bound ComfyUI and the release disagree on.

    Neither bound binds on any graph this repo ships -- the append node puts
    every reference at a 2048 short edge, which is above the release's floor and
    below ComfyUI's ceiling until roughly 3:1. This exists to say WHEN that
    stops being true.

    **Only meaningful under `image_policy='comfy'`.** The two warnings below
    describe what happens when the conditioner hands the still on untouched and
    lets whatever processor the CLIP carries resize it. Under `release` or
    `encoder` the conditioner has already applied that policy's own floor and
    ceiling before the VAE, so both sentences would be false and the caller
    does not ask.
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
