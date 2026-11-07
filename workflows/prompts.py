"""The prompt bank as the ONE place prompt text lives, and the lookup that
stamps a render's prompt into its records.

Until 2026-09-03 the generator held every shipped prompt as a string
constant, `prompt_bank/` held a second population that had never rendered,
and nothing joined them: the catalogue derived scenes from the graphs and
named them after the constants, the bank graded its own files, and a record
of a render named its workflow file and nothing about what was rendered.
Every real-activation Sol number in the repo came from one scene as a
result, and nobody could tell from a record which. The owner's call: one
source of truth for prompt text, adaptable where the text allows, and the
exact prompt in every record.

So: `prompt_bank/<id>.txt` is the text, `prompt_bank/bank.json` is the
manifest (mode, frames, donor, brief), and the generator's constants are
now `text("<id>")`. The constant NAMES stay, because the bench scripts and
the catalogue read them; only the literal moved. A prompt that is not in
the bank cannot be shipped, which is the invariant this module exists for.

**That invariant covers COMPOSED prompts since later the same day.** The
ref2va arms are built from role tables by `build_workflows._ref_prompt` and
the two keyframe defaults resolve a duration into their Part One line, so
they arrive as output rather than as an id and had stayed outside the bank
-- the catalogue could only name them `derived:<graph>`. The generator now
looks its composed text up through `identify` and ships the bank's copy,
refusing to build otherwise, so `describe` resolves an id for every prompt
the repo renders rather than for most of them.

`describe(graph)` is the record side: given an API graph it returns the
bank id (or None for a foreign prompt), the text's sha256, the length,
canvas and seed, so `bench/run_graph_arms.py` rows and the Sol route
record's render row say what was rendered rather than only which file.

`carriers(graph)` is the read side for GRADERS: which nodes of a graph carry
prompt text and what the encoder reads from each (a song node's template
expanded through its Prompt Lists). `h3_config.PROMPT_INPUTS` is the one
registry of carrier classes; nothing else keeps a list of them.

No torch, no ComfyUI: importable from the generator and the bench.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h3_config as _cfg

REPO = Path(__file__).resolve().parent.parent
BANK = REPO / "prompt_bank"
MANIFEST = BANK / "bank.json"

def text(prompt_id: str) -> str:
    """The bank text for `prompt_id`, STRIPPED of leading and trailing
    whitespace. Raises if the file is absent, so a generator naming a prompt
    that does not exist fails at import.

    Stripped since 2026-09-18 (the owner: "avoid footguns"). The files end in
    a newline, as text files do, and this used to hand that over verbatim, so
    the newline was baked into some shipped graphs and passed on by runners.
    One character in a prompt is a different sample: on 2026-09-17 a trailing
    newline made two "identical" arms different renders and a day of stacks
    compared them as one. Stripping here, at the one door every caller comes
    through, means no caller has to remember. `identify` and `sha256` already
    right-strip, so ids and hashes do not move."""
    path = BANK / f"{prompt_id}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"prompt_bank/{prompt_id}.txt does not exist; "
                                f"every shipped prompt must live in the bank")
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def entries() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["prompts"]


def entry(prompt_id: str) -> dict | None:
    return next((e for e in entries() if e["id"] == prompt_id), None)


def sha256(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _by_text() -> dict[str, str]:
    """Normalised text -> id. Trailing whitespace is the one thing allowed to
    differ, because a widget round-trip can strip it."""
    out = {}
    for e in entries():
        p = BANK / f"{e['id']}.txt"
        if p.is_file():
            out[p.read_text(encoding="utf-8").rstrip()] = e["id"]
    ph3_dir = BANK / "p_h3_image"
    if ph3_dir.is_dir():
        for p in sorted(ph3_dir.glob("*.txt")):
            out[p.read_text(encoding="utf-8").rstrip()] = f"p_h3_image/{p.stem}"
    return out


def identify(prompt_text: str) -> str | None:
    """The bank id whose text this is, or None for a prompt not in the bank."""
    return _by_text().get(prompt_text.rstrip())


def describe(graph: dict) -> dict:
    """What an API graph renders: prompt id and hash, length, canvas, seed.

    Reads the graph, never the bank's opinion of it: `prompt_id` is None and
    `prompt_text` is excluded to keep captures and render records private.
    Keys are always present; unknown values are None."""
    out = {"prompt_id": None, "prompt_sha256": None,
           "length": None, "canvas": None, "seed": None}
    if not isinstance(graph, dict):
        return out
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        ct, inputs = node.get("class_type"), node.get("inputs") or {}
        # followed through a link, so a prompt a Masked Prompt node writes is identified too
        t = _follow(inputs.get(_cfg.PROMPT_INPUTS[ct]), graph)[0] if ct in _cfg.PROMPT_INPUTS else None
        if isinstance(t, str):
            out["prompt_sha256"] = sha256(t.rstrip())
            out["prompt_id"] = identify(t)
        elif ct in ("MiniMaxH3PH3ImageConditioning", "PH3ImageConditioning"):
            t = _follow(inputs.get("prompt"), graph)[0]
            if isinstance(t, str):
                out["prompt_sha256"] = sha256(t.rstrip())
                out["prompt_id"] = identify(t)
            out["length"] = 1
            w, h = inputs.get("width", 0), inputs.get("height", 0)
            has_images = any(k.startswith("images.image_") and inputs[k] is not None for k in inputs)
            if ct == "MiniMaxH3PH3ImageConditioning" and inputs.get("canvas", "auto") == "auto" and has_images:
                out["canvas"] = "auto"
            else:
                out["canvas"] = f"{w}x{h}" if (w and h) else "auto"
        elif ct == "MiniMaxH3Resolution":
            out["length"] = inputs.get("length")
            shape = inputs.get("shape")
            res = inputs.get(f"shape.{shape}_resolution") if shape else None
            out["canvas"] = res.split()[0] if isinstance(res, str) else res
        elif ct == "RandomNoise":
            out["seed"] = inputs.get("noise_seed")
    return out


# ---------------------------------------------------------------------------
# What a graph's encoder reads: carriers
# ---------------------------------------------------------------------------

# The template grammar the song node and Fill Prompt Lists use. These three are
# COPIES, because `prompt_lists.py` and `loop_plan.py` import ComfyUI and this
# module must not; `bench/check_prompt_lists.py` pins each against the original
# and goes red if either side moves. Behaviour copied, not reinvented:
# `prompt_lists.PLACEHOLDER` / `parse_values` and `loop_plan.BLOCK_LINE`.
PLACEHOLDER = re.compile(r"__([A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*)__")
BLOCK_LINE = re.compile(r"^---(.*)$")


def placeholders(text: str) -> list[str]:
    """The distinct placeholder names in `text`, in order of first appearance."""
    seen: list[str] = []
    for m in PLACEHOLDER.finditer(text):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def parse_values(text: str) -> tuple[str, ...]:
    """One value per line; blank lines and lines starting with # are skipped."""
    out = []
    for line in (text or "").replace("\r\n", "\n").split("\n"):
        value = line.strip()
        if value and not value.startswith("#"):
            out.append(value)
    return tuple(out)


def split_blocks(text: str) -> list[str]:
    """The prompt blocks of a template: the whole text when it has no `---` line,
    else one block per `--- label` line. Lenient where the node raises (an empty
    or unlabelled block is dropped here), because a grader reports on what is
    there and the node refuses the rest at run time."""
    lines = (text or "").replace("\r\n", "\n").split("\n")
    if not any(BLOCK_LINE.match(line.strip()) for line in lines):
        return [text.strip()] if (text or "").strip() else []
    blocks: list[str] = []
    current: list[str] | None = None
    for raw in lines:
        if BLOCK_LINE.match(raw.strip()):
            if current is not None and "\n".join(current).strip():
                blocks.append("\n".join(current).strip())
            current = []
        elif current is not None:
            current.append(raw)
    if current is not None and "\n".join(current).strip():
        blocks.append("\n".join(current).strip())
    return blocks


@dataclass(frozen=True)
class Carrier:
    """One node of a graph that carries prompt text, and what the encoder reads.

    `text` is what is stored in the node: a template for a song node. `texts`
    is what the encoder reads: `(text,)` for an ordinary carrier, and for a
    template every block filled so that each value of each list appears in at
    least one text (the others held at their first value), which covers every
    value without a combinatorial product. `unresolved` names a placeholder
    with no typed values to fill it (a wildcard-file list, or none chained);
    its token is left in the text. `missing` names a placeholder with no list
    chained at all and `unused` a chained list no block uses; the node refuses
    both at run time. `note` says why `texts` is empty or partial. `template`
    is True when `text` is a template and `texts` its expansion.
    """
    node_id: str
    cls: str
    text: str
    texts: tuple[str, ...]
    window_frames: int | None = None
    unresolved: tuple[str, ...] = ()
    note: str = ""
    template: bool = False
    missing: tuple[str, ...] = ()
    unused: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def _masked_text():
    """`masked_prompt_text.py` from the repo root, loaded by path: it is the
    one place the masked lane's sentences live, and it imports nothing."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("masked_prompt_text", REPO / "masked_prompt_text.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assembled(ins: dict, graph: dict) -> str | None:
    """What a Masked Prompt node (`masked_prompt.py`) writes: its widget values,
    and `replace` and `motion_reference` read off the Masked Source wired into
    its `source`, as the node reads them at run time. None when a value is
    itself a link, or the combination is one the node refuses."""
    m = _masked_text()
    link = ins.get("source")
    wired = isinstance(link, list) and len(link) == 2 and str(link[0]) in graph
    up = (graph[str(link[0])].get("inputs", {}) or {}) if wired else None
    try:
        return m.assemble(ins.get("subject", m.SUBJECT_PERSON), ins.get("voice", m.VOICE_MAIN),
                          ins.get("picture_gives", m.GIVES_FOLLOW), ins.get("add_to_shot", ""),
                          up.get("replace", m.REPLACE_WHOLE) if wired else None,
                          up.get("motion_reference", m.MOTION_NONE) if wired else None)
    except (ValueError, TypeError, AttributeError):
        return None


def _follow(value, graph: dict, *, hops: int = 8):
    """Follow a two-item link from a prompt input to the node that supplies it.
    Returns (text, source node dict or None)."""
    seen: set[str] = set()
    source = None
    while isinstance(value, list) and len(value) == 2 and hops:
        sid = str(value[0])
        if sid in seen or sid not in graph:
            return None, None
        seen.add(sid)
        source = graph[sid]
        ins = source.get("inputs", {}) or {}
        # a frontend string primitive carries the text as `value`; Fill Prompt
        # Lists carries the template as `prompt`; a Masked Prompt node writes it
        if source.get("class_type") == _cfg.MASKED_PROMPT_NODE:
            value = _assembled(ins, graph)
        else:
            value = ins.get("value", ins.get("text", ins.get("prompt")))
        hops -= 1
    return (value, source) if isinstance(value, str) else (None, None)


def _list_values(link, graph: dict) -> dict[str, tuple[str, ...] | None]:
    """{placeholder name: typed values, or None when the list is a wildcard file}
    for the Prompt List nodes chained from `link`, newest first."""
    out: dict[str, tuple[str, ...] | None] = {}
    seen: set[str] = set()
    while isinstance(link, list) and len(link) == 2 and str(link[0]) in graph:
        nid = str(link[0])
        if nid in seen:
            break
        seen.add(nid)
        node = graph[nid]
        ins = node.get("inputs", {}) or {}
        if node.get("class_type") == "MiniMaxH3PromptList":
            name = str(ins.get("name", "")).strip()
            if len(name) > 4 and name.startswith("__") and name.endswith("__"):
                name = name[2:-2]
            if name and name not in out:
                typed = ins.get("source") == "typed" and isinstance(ins.get("source.values"), str)
                out[name] = parse_values(ins["source.values"]) if typed else None
        link = ins.get("lists")
    return out


def _expand(blocks: list[str], lists: dict[str, tuple[str, ...] | None]):
    """Every block filled so each value of each list is read at least once.
    Returns (texts, unresolved, missing, unused); see `Carrier`."""
    texts: list[str] = []
    unresolved: list[str] = []
    missing: list[str] = []
    used: set[str] = set()

    def fill(block: str, chosen: dict[str, str]) -> str:
        return PLACEHOLDER.sub(lambda m: chosen.get(m.group(1), m.group(0)), block)

    for block in blocks:
        names = placeholders(block)
        used.update(names)
        first = {}
        for n in names:
            vals = lists.get(n)
            if vals:
                first[n] = vals[0]
            elif n not in lists:
                if n not in missing:
                    missing.append(n)
            elif n not in unresolved:
                unresolved.append(n)
        variants = [fill(block, first)]
        for n in names:
            for v in (lists.get(n) or ())[1:]:
                variants.append(fill(block, {**first, n: v}))
        for t in variants:
            if t not in texts:
                texts.append(t)
    unused = sorted(set(lists) - used)
    return tuple(texts), tuple(unresolved), tuple(missing), tuple(unused)


def carriers(graph: dict) -> list[Carrier]:
    """Every prompt carrier in an API graph, with what the encoder reads from it.

    The one answer to "what text does this graph's encoder read", for the
    graders and the catalogue. Classes come from `h3_config.PROMPT_INPUTS`; a
    prompt that arrives by link is followed to its string source. A template
    carrier (`h3_config.PROMPT_TEMPLATE_CARRIERS`, or a prompt fed by a Fill
    Prompt Lists node) is expanded through its chained lists.
    """
    out: list[Carrier] = []
    if not isinstance(graph, dict):
        return out
    for nid, node in graph.items():
        if not isinstance(node, dict):
            continue
        cls = node.get("class_type")
        field = _cfg.PROMPT_INPUTS.get(cls)
        if field is None:
            continue
        ins = node.get("inputs", {}) or {}
        text, source = _follow(ins.get(field), graph)
        if text is None:
            out.append(Carrier(str(nid), cls, "", (), note="the prompt is linked and its string source could not be resolved"))
            continue
        feeder = source is not None and source.get("class_type") == "MiniMaxH3FillPromptLists"
        if cls in _cfg.PROMPT_TEMPLATE_CARRIERS or feeder:
            lists_link = ins.get("lists") if cls in _cfg.PROMPT_TEMPLATE_CARRIERS else (source.get("inputs", {}) or {}).get("lists")
            texts, unresolved, missing, unused = _expand(split_blocks(text), _list_values(lists_link, graph))
            frames = ins.get("window_frames")
            out.append(Carrier(str(nid), cls, text, texts,
                               frames if isinstance(frames, int) and not isinstance(frames, bool) else None,
                               unresolved, template=True, missing=missing, unused=unused))
        else:
            out.append(Carrier(str(nid), cls, text, (text,)))
    return out
