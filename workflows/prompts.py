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
now `text("<id>")`. The constant NAMES stay, because five bench scripts and
the catalogue import them; only the literal moved. A prompt that is not in
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

No torch, no ComfyUI: importable from the generator, the bench and the
node pack alike.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BANK = REPO / "prompt_bank"
MANIFEST = BANK / "bank.json"

# Node class -> the input carrying the prompt text, for `describe`. The
# catalogue keys on the same set (`bench/build_prompt_catalogue.py::CONDITIONERS`).
CONDITIONERS = {"MiniMaxH3Conditioning": "prompt"}


def text(prompt_id: str) -> str:
    """The bank text for `prompt_id`, verbatim. Raises if the file is absent,
    so a generator naming a prompt that does not exist fails at import."""
    path = BANK / f"{prompt_id}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"prompt_bank/{prompt_id}.txt does not exist; "
                                f"every shipped prompt must live in the bank")
    return path.read_text(encoding="utf-8")


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
    return out


def identify(prompt_text: str) -> str | None:
    """The bank id whose text this is, or None for a prompt not in the bank."""
    return _by_text().get(prompt_text.rstrip())


def describe(graph: dict) -> dict:
    """What an API graph renders: prompt id and hash, length, canvas, seed.

    Reads the graph, never the bank's opinion of it: `prompt_id` is None and
    `prompt_text` is carried in full when the text is not a bank entry, so
    a foreign or hand-edited prompt is recorded rather than lost. Keys are
    always present; unknown values are None."""
    out = {"prompt_id": None, "prompt_sha256": None, "prompt_text": None,
           "length": None, "canvas": None, "seed": None}
    if not isinstance(graph, dict):
        return out
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        ct, inputs = node.get("class_type"), node.get("inputs") or {}
        if ct in CONDITIONERS and isinstance(inputs.get(CONDITIONERS[ct]), str):
            t = inputs[CONDITIONERS[ct]]
            out["prompt_sha256"] = sha256(t.rstrip())
            out["prompt_id"] = identify(t)
            out["prompt_text"] = None if out["prompt_id"] else t
        elif ct == "MiniMaxH3Resolution":
            out["length"] = inputs.get("length")
            shape = inputs.get("shape")
            res = inputs.get(f"shape.{shape}_resolution") if shape else None
            out["canvas"] = res.split()[0] if isinstance(res, str) else res
        elif ct == "RandomNoise":
            out["seed"] = inputs.get("noise_seed")
    return out
