"""Prompt lists: fill `__name__` placeholders, one value per use, no repeat before a list is used up.

`docs/h3/audio_freeze.md` owns the lane that uses these first (the song node's
`lists` input); nothing here is specific to audio. The owner's rules
(2026-09-14), and where each lives:

- **A placeholder is `__name__`.** Not `{name}`, which the frontend's dynamic
  prompts rewrite before a node sees it, and not `[name]` or `<name>`, which
  H3 prompts already use (`[Shot 1]`, `<Subject 1>`). A name is letters and
  digits with single `_`, `-` or `/` separators. `PLACEHOLDER`.
- **A list advances only when a use of the prompt needs it**, and the same
  placeholder twice in one text takes the same value. `resolve_texts`. What a
  use is belongs to the caller: a window, or a timeline entry of the song node.
- **`shuffled` never repeats a value before the list is used up.** Each pass
  through the list is a fresh shuffle, never the same order as the pass
  before, and with three or more values the first value of a pass is never
  the last of the one before. With more values than uses, no value repeats at
  all. `next_pass`. `in_order` runs the list in order, pass after pass;
  `random` draws independently and may repeat, and is only ever chosen.
- **`shuffle` picks which shuffles, and nothing moves it.** The same number
  gives the same sequence on every render. It is not named `seed` because the
  frontend draws a control that changes an INT named `seed` after each queue,
  and a filled-in prompt that changes on every queue re-renders every window
  resume could have kept (owner, 2026-09-14).

**Every list comes from a list node.** A placeholder with no Prompt List of its
name is refused, and so is a list a loop connects that none of its texts uses:
both are a typo in a name. (Until 2026-09-14 a placeholder with no list read
`name.txt` or `name.json` from the wildcards folder by itself; the owner cut
that fallback with its path and named-list lookups.)

**Every loop fills the same way.** A node that loops over windows inside
itself takes a `lists` input and calls `fill_windows` while it plans.
`bench/check_prompt_lists.py` fails a node that writes windows through
`loop_output` or resumes through `loop_resume` without both. A graph that loops
by chaining nodes, or renders one clip per queue, uses
`MiniMaxH3FillPromptLists`, whose `index` names the use. Value N of a list is a
function of its values, order, shuffle and N alone, so both routes give the
same values.

**Wildcard files** live in `wildcards/` under ComfyUI's input directory
(`register_wildcards_folder`, the `--input-directory` override included), and
`extra_model_paths.yaml` can add more under the key `wildcards`. A list node
with the `file` source reads one: a `.txt` holds one value per line (blank
lines and `#` lines skipped), a `.json` holds one list of strings. The node
fingerprints the file, so editing it runs the node and everything after it
again.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass

from comfy_api.latest import io

logger = logging.getLogger(__name__)

PLACEHOLDER = re.compile(r"__([A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*)__")
ORDERS = ("shuffled", "in_order", "random")
WILDCARD_EXTENSIONS = (".txt", ".json")
#: Attempts at a pass that satisfies both rules before the deterministic
#: fallback. Reasoned: a random shuffle of three or more values meets them
#: far more often than not, so this bounds a loop that in practice runs once.
PASS_ATTEMPTS = 64

H3PromptLists = io.Custom("H3_PROMPT_LISTS")


@dataclass(frozen=True)
class PromptList:
    name: str
    values: tuple[str, ...]
    order: str
    shuffle: int
    source: str  # "typed", or the wildcard file the values came from


def clean_name(raw: str) -> str:
    """A list name with or without its surrounding `__`, checked against `PLACEHOLDER`."""
    name = (raw or "").strip()
    if len(name) > 4 and name.startswith("__") and name.endswith("__"):
        name = name[2:-2]
    if not PLACEHOLDER.fullmatch(f"__{name}__"):
        raise ValueError(f"{raw!r} is not a list name: use letters and digits with single _, - or / "
                         "between them, and write it in the prompt as __name__")
    return name


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


def values_from_file(path: str) -> tuple[str, ...]:
    """A wildcard file's values: a `.txt` one per line, a `.json` one list of strings."""
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not all(isinstance(v, str) for v in data):
            raise ValueError(f"{os.path.basename(path)}: a JSON wildcard file holds one list of strings")
        values = tuple(v.strip() for v in data if v.strip())
    else:
        with open(path, encoding="utf-8") as f:
            values = parse_values(f.read())
    if not values:
        raise ValueError(f"{os.path.basename(path)} holds no values")
    return values


def next_pass(rng, values: tuple[str, ...], previous: list[int] | None) -> list[int]:
    """One shuffled pass over `values`, as indices, obeying the two rules in the module docstring."""
    k = len(values)
    order = list(range(k))
    if k == 1:
        return order
    if previous is None:
        rng.shuffle(order)
        return order
    for _ in range(PASS_ATTEMPTS):
        perm = order[:]
        rng.shuffle(perm)
        if perm == previous:
            continue
        if k >= 3 and values[perm[0]] == values[previous[-1]]:
            continue
        return perm
    # Deterministic fallback: the previous pass rotated by one differs from it,
    # and with three or more values its first index is not the previous last.
    return previous[1:] + previous[:1]


class ListSequence:
    """The values a list gives, in order of use; a function of (name, values, order, shuffle) alone."""

    def __init__(self, plist: PromptList):
        if plist.order not in ORDERS:
            raise ValueError(f"unknown order {plist.order!r}; one of {ORDERS}")
        if not plist.values:
            raise ValueError(f"list {plist.name!r} holds no values")
        self.plist = plist
        # the same string the list node's `seed` fed until 2026-09-14, so a
        # renamed input gives the values it gave before
        self._rng = random.Random(f"{int(plist.shuffle)}:{plist.name}")
        self._drawn: list[int] = []
        self._last_pass: list[int] | None = None

    def index(self, n: int) -> int:
        while len(self._drawn) <= n:
            k = len(self.plist.values)
            if self.plist.order == "in_order":
                self._drawn.extend(range(k))
            elif self.plist.order == "random":
                self._drawn.append(self._rng.randrange(k))
            else:
                self._last_pass = next_pass(self._rng, self.plist.values, self._last_pass)
                self._drawn.extend(self._last_pass)
        return self._drawn[n]

    def value(self, n: int) -> str:
        return self.plist.values[self.index(n)]


def as_dict(chain) -> dict[str, PromptList]:
    """The lists a `lists` input carries, by name."""
    return {pl.name: pl for pl in (chain or ())}


def _require_lists(texts: list[str], lists) -> None:
    missing = sorted({n for t in texts for n in placeholders(t)} - set(lists))
    if missing:
        raise ValueError(f"no list for {', '.join('__' + n + '__' for n in missing)}: connect a Prompt List "
                         f"named {', '.join(repr(n) for n in missing)} to `lists`")


def resolve_texts(texts: list[str], lists: dict[str, PromptList]) -> tuple[list[str], list[dict[str, str]]]:
    """Fill every text's placeholders; a list advances once for each text that uses it.

    Returns the filled texts and, per text, the value each name took. Raises
    naming every placeholder without a list.
    """
    _require_lists(texts, lists)
    sequences = {name: ListSequence(pl) for name, pl in lists.items()}
    uses = dict.fromkeys(lists, 0)
    filled, picks = [], []
    for text in texts:
        chosen = {}
        for name in placeholders(text):
            chosen[name] = sequences[name].value(uses[name])
            uses[name] += 1
        filled.append(PLACEHOLDER.sub(lambda m: chosen[m.group(1)], text))
        picks.append(chosen)
    return filled, picks


def report_lines(picks: list[dict[str, str]], first: int = 1) -> list[str]:
    """One line per text that used a list, numbered from `first`, as reports and the log show them."""
    return [f"[{first + i}] " + ", ".join(f"__{name}__ = {value!r}" for name, value in chosen.items())
            for i, chosen in enumerate(picks) if chosen]


def fill_windows(texts: list[str], chain) -> tuple[list[str], list[str]]:
    """What a loop node calls while it plans: each use's text filled, and the report lines.

    `texts` is one text per use, in order (a window, or a timeline entry).
    Call it before anything is keyed or encoded, so resume and any encode
    cache see the text a window renders. Raises on a placeholder with no list
    and on a connected list no text uses. The lines are logged here and belong
    in the node's report too.
    """
    lists = as_dict(chain)
    used = {n for t in texts for n in placeholders(t)}
    unused = sorted(set(lists) - used)
    if unused:
        raise ValueError(f"the list(s) {', '.join(repr(n) for n in unused)} are connected but no prompt uses "
                         f"{', '.join('__' + n + '__' for n in unused)}")
    filled, picks = resolve_texts(texts, lists)
    lines = report_lines(picks)
    for line in lines:
        logger.info("[h3]   %s", line)
    return filled, lines


def fill_at(text: str, lists: dict[str, PromptList], use: int) -> tuple[str, dict[str, str]]:
    """`text` with each placeholder at its list's use `use`, counted from 0.

    Window N of a loop whose every text uses a name takes that name's use
    N - 1, so this agrees with `resolve_texts` there.
    """
    _require_lists([text], lists)
    chosen = {name: ListSequence(lists[name]).value(int(use)) for name in placeholders(text)}
    return PLACEHOLDER.sub(lambda m: chosen[m.group(1)], text), chosen


def register_wildcards_folder() -> str | None:
    """`wildcards/` under the input directory, registered with core as the folder type `wildcards`.

    Core's registration keeps the extension set of a folder type that already
    exists (`folder_paths.add_model_folder_path`), and a new one starts empty,
    which lists every file; so the set is written here either way.
    """
    import folder_paths
    path = os.path.join(folder_paths.get_input_directory(), "wildcards")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        logger.warning("[h3] could not create the wildcards folder %s: %s", path, exc)
    folder_paths.add_model_folder_path("wildcards", path, is_default=True)
    paths = folder_paths.folder_names_and_paths["wildcards"][0]
    folder_paths.folder_names_and_paths["wildcards"] = (paths, set(WILDCARD_EXTENSIONS))
    return path


def find_wildcard(relative: str) -> str | None:
    """A wildcard file's full path, or None (also when the folder was never registered)."""
    import folder_paths
    try:
        return folder_paths.get_full_path("wildcards", relative)
    except KeyError:
        return None


def _wildcard_files() -> list[str]:
    try:
        import folder_paths
        return folder_paths.get_filename_list("wildcards")
    except (ImportError, KeyError):
        return []


class MiniMaxH3PromptList(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PromptList",
            display_name="MiniMax H3 Prompt List",
            category="MiniMaxH3",
            description=(
                "A list of values for a __name__ placeholder: typed one per line, or read from a file in "
                "the wildcards folder under ComfyUI's input directory. Chain several, one per name, into "
                "a loop node's `lists`. shuffled uses every value once before any repeats and reshuffles "
                "each pass; in_order runs the list in order; random may repeat."
            ),
            inputs=[
                io.String.Input("name", default="subject",
                                tooltip="The placeholder this list fills, written in the prompt as __name__."),
                io.DynamicCombo.Input(
                    "source",
                    options=[
                        io.DynamicCombo.Option("typed", [
                            io.String.Input("values", multiline=True, default="",
                                            tooltip="One value per line; blank lines and # lines are skipped."),
                        ]),
                        io.DynamicCombo.Option("file", [
                            io.Combo.Input("wildcard", options=_wildcard_files(),
                                           tooltip=("A .txt (one value per line) or .json (one list of "
                                                    "strings) in the wildcards folder. A file added while "
                                                    "ComfyUI runs appears after refreshing node definitions "
                                                    "(R in the editor), no restart; on a network share it can "
                                                    "take a moment longer.")),
                        ]),
                    ],
                    tooltip="Where the values come from."),
                io.Combo.Input("order", options=list(ORDERS), default="shuffled",
                               tooltip=("shuffled: every value once before any repeats, a fresh order each "
                                        "pass. in_order: the list in order, pass after pass. random: an "
                                        "independent draw each time; repeats allowed.")),
                io.Int.Input("shuffle", default=0, min=0, max=0xffffffffffffffff,
                             tooltip=("Which shuffles, for shuffled and random. The same number gives the same "
                                      "sequence on every render; change it by hand for a different one.")),
                H3PromptLists.Input("lists", optional=True, tooltip="Lists chained before this one."),
            ],
            outputs=[H3PromptLists.Output(display_name="lists")],
        )

    @classmethod
    def fingerprint_inputs(cls, source=None, **kwargs):
        # an edited wildcard file changes no input; its time and size do
        rel = source.get("wildcard") if isinstance(source, dict) else kwargs.get("source.wildcard")
        full = find_wildcard(rel) if isinstance(rel, str) and rel else None
        if full is None:
            return None
        st = os.stat(full)
        return (full, st.st_mtime_ns, st.st_size)

    @classmethod
    def execute(cls, name, source, order, shuffle, lists=None) -> io.NodeOutput:
        name = clean_name(name)
        if order not in ORDERS:
            raise ValueError(f"unknown order {order!r}; one of {ORDERS}")
        # A DynamicCombo arrives as a nested dict (the selection under its own
        # id, the option's inputs beside it) or, selection only, as a string.
        choice = source if isinstance(source, str) else source.get("source")
        option = {} if isinstance(source, str) else source
        if choice == "typed":
            values = parse_values(option.get("values", ""))
            where = "typed"
            if not values:
                raise ValueError(f"list {name!r} holds no values")
        elif choice == "file":
            rel = option.get("wildcard") or ""
            full = find_wildcard(rel) if rel else None
            if full is None:
                raise ValueError(f"list {name!r}: no wildcard file {rel!r} in the wildcards folder")
            values = values_from_file(full)
            where = rel
        else:
            raise ValueError(f"unknown source {choice!r}")
        chain = tuple(lists or ())
        if any(pl.name == name for pl in chain):
            raise ValueError(f"two lists are named {name!r}")
        plist = PromptList(name, values, order, int(shuffle), where)
        logger.info("[h3] prompt list __%s__: %d value(s) from %s, %s, shuffle %d",
                    name, len(values), where, order, int(shuffle))
        return io.NodeOutput(chain + (plist,))


class MiniMaxH3FillPromptLists(io.ComfyNode):
    """Placeholders for a graph that is not one loop node; the module docstring says how the routes agree."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FillPromptLists",
            display_name="MiniMax H3 Fill Prompt Lists",
            category="MiniMaxH3",
            description=(
                "Fills __name__ placeholders in a prompt for any prompt input: a chain of window nodes, or "
                "one clip per queue. index is which use of each list, from 1, so window N of a chain takes "
                "index N. A loop node with its own `lists` input fills its windows itself."
            ),
            inputs=[
                io.String.Input("prompt", multiline=True, default="",
                                tooltip="Text with __name__ placeholders."),
                # Fixed by default (owner, 2026-09-14): a prompt that moves on
                # its own after a queue is the footgun the list node's rename
                # removed. Set the control to increment to walk the lists one
                # queue at a time.
                io.Int.Input("index", default=1, min=1, max=1000000,
                             control_after_generate=io.ControlAfterGenerate.fixed,
                             tooltip=("Which use of each list, from 1. Wire a window number in, or set the "
                                      "control to increment so repeated queues take the next values.")),
                H3PromptLists.Input("lists", optional=True,
                                    tooltip="Prompt List nodes, one per name the prompt uses."),
            ],
            outputs=[io.String.Output(display_name="prompt"),
                     io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, prompt, index, lists=None) -> io.NodeOutput:
        text, chosen = fill_at(prompt, as_dict(lists), int(index) - 1)
        lines = report_lines([chosen], first=int(index))
        for line in lines:
            logger.info("[h3]   %s", line)
        return io.NodeOutput(text, "\n".join(lines) or "no placeholders")
