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
    _STRUCTURE_PROBES, _audio_sections_optional)

# Every node that carries a prompt into the encoder. `MiniMaxH3Conditioning`
# is this repo's own, and the fl2va path moved onto it on 2026-08-21; a file
# that knows only the first name reports "nothing to grade" over every graph
# this repo now ships and reads as a clean pass.
PROMPT_NODES = ("MiniMaxH3ReferenceToVideo", "MiniMaxH3Conditioning")
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


def guide_for(inputs: dict) -> str:
    """"ref" when the node wires reference labels, else "base"."""
    return "ref" if any(k.startswith(REF_SOCKET_PREFIXES) for k in inputs) \
        else "base"

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


def grade(node: dict, graph: dict, stem: str = "") -> list[tuple[str, str]]:
    """Mechanical rules only. Every one is decidable from the prompt + sockets."""
    out = []
    ins = node["inputs"]
    prompt = ins.get("prompt", "")
    guide = guide_for(ins)
    sections = REF_SECTIONS if guide == "ref" else BASE_SECTIONS
    main_field = MAIN_FIELD[guide]
    sec = split_sections(prompt, sections)
    expected = wired_labels(ins)
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
    for lab in expected:
        if lab not in named:
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

    # base-en's Part One. The keyframe tasks each require an alignment
    # instruction as the FIRST line, before the core fields (base-en.txt:19,
    # :25, :29); T2VA "has no image-alignment instruction and begins directly
    # with the three core fields" (base-en.txt:14). Both halves are graded,
    # because a rule with only one side is satisfied by deleting the text.
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
    return out


def price(node: dict, graph: dict) -> list[str]:
    ins = node["inputs"]
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

    size_mode = ins.get("ref_image_size", "match")
    ref_total = 0
    for key, link in ins.items():
        if not key.startswith("ref_images.") or not isinstance(link, list):
            continue
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
        n = sum(1 for k, v in ins.items() if k.startswith(prefix) and v is not None)
        if n:
            unpriced.append(f"{n} {kind_}")
    if unpriced:
        lines.append(f"  NOT COUNTED: {', '.join(unpriced)} reference(s). A "
                     f"video reference alone measured 52,020 rows at 960x544 / "
                     f"345f, so the total below is a FLOOR, not a budget.")

    prompt = ins.get("prompt", "")
    tt, kind = text_tokens(prompt)
    twin = ref_total + 100 if ref_total else 0
    lines.append(f"  text      {tt:>8,}  prompt tokens ({kind})"
                 + (f" + ~{twin:,} vision blocks" if twin else ""))

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
        _comfy = Path.home() / "ComfyUI"
        if sys.path[0] != str(_comfy):
            sys.path.insert(0, str(_comfy))
        from comfy_extras.nodes_minimax_h3 import temporal_shape
        audio_rows = temporal_shape(snapped)[2] * 2
    except Exception:
        lines.append("  NOT COUNTED: target audio rows -- ComfyUI is not "
                     "importable from here, so `temporal_shape` could not be "
                     "read. That segment is ~1,206 rows at 362 frames and is "
                     "always present, so the total below is short by about "
                     "that much.")
    if audio_rows:
        lines.append(f"  audio     {audio_rows:>8,}  target audio rows "
                     f"(always present, soundtrack or not)")

    total = video + ref_total + tt + twin + audio_rows
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
