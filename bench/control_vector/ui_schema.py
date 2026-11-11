"""The studio's description of the CLI: every method and argument, read off the parser.

The page draws its controls from `method_schema()` and posts back a plain `{dest: value}` map;
`argv_for()` turns that map into the argument list the command line would have been given, and
the CLI's own parser reads it. So the parser is the one definition of a method's arguments, their
choices and their defaults: nothing here, in the server or on the page repeats one.

Two things live here that the parser cannot say, and neither is an option or a default:

- `SERVER_SET`: where the studio writes. The page never shows these arguments.
- `OUTPUT_PATHS`: the arguments that name a file or folder a method writes. The page shows them,
  and the server takes one only inside the studio's own two folders.
- `WIDGETS` and `PRIMARY`: how a control is drawn (a text area, a picture drop, a vector picker)
  and which controls sit above "More options". `bench/check_control_vector_studio.py` holds every
  name in them to an argument the parser really has.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
from typing import Any

import control_vectors as cv
import h3_rules
import h3_spans

from .cli import build_parser

#: Subcommands the page does not offer: `ui` is the studio itself.
SKIP_METHODS: frozenset[str] = frozenset({"ui"})

#: `{method: {dest: state attribute}}`: arguments the studio sets from its own folders. The page
#: does not show them and `argv_for` refuses them from a request.
SERVER_SET: dict[str, dict[str, str]] = {
    "from-prompt": {"out_dir": "vec_dir"},
    "build": {"out_dir": "vec_dir"},
    "make-media": {"out_dir": "vec_dir"},
    "words": {"out_dir": "vec_dir"},
    "sculpt": {"out_dir": "vec_dir"},
    "rebuild": {"vec_dir": "vec_dir", "toml_dir": "toml_dir"},
}

#: `{method: (dest, ...)}`: arguments that name where a method writes. From the page a value is
#: taken only if it lies inside the studio's vector folder or pair folder (`ui_server.run_method`);
#: to write anywhere else, use the command line or start the studio with `--vec-dir`/`--toml-dir`.
#: `bench/check_control_vector_studio.py` fails on a path-shaped argument that is in neither this
#: table nor `SERVER_SET`.
OUTPUT_PATHS: dict[str, tuple[str, ...]] = {
    "from-prompt": ("toml_out",),
    "build": ("toml_out", "test_prompt_out"),
    "from-md": ("out", "test_prompt_out"),
    "make": ("out",),
    "grade": ("out",),
    "generate-pairs": ("out_md", "out_toml"),
    "build-j-lens": ("out",),
}

#: `{dest: widget}`: how the page draws an argument, wherever a method has it. Presentation only.
WIDGETS: dict[str, str] = {
    "target_prompt": "textarea",
    "words": "textarea",
    "prompt": "textarea",
    "scene_prompt": "textarea",
    "concept": "textarea",
    "image_a": "image",
    "image_b": "image",
    "vector": "vector",
    "vectors": "vectors",
    "hold": "vectors",
    "source": "pairfile",
    "pairs": "pairfile",
    # A named mask, or one or more prompt fields ticked: the node's `mask` and `fields` in one control.
    "mask": "mask",
}

#: `{method: [dest, ...]}`: the arguments shown above "More options", in this order. A method not
#: named here shows its required and positional arguments first.
PRIMARY: dict[str, tuple[str, ...]] = {
    "from-prompt": ("target_prompt", "target_prompt_file", "words", "name", "families"),
    "build": ("source", "name", "families", "pooling"),
    "make": ("pairs", "output_name", "families", "pooling"),
    "make-media": ("image_a", "image_b", "output_name", "scene_prompt", "families"),
    "words": ("pos_words", "against_words", "hold_words", "name", "families"),
    "sculpt": ("vectors", "modes", "strengths", "name", "families"),
    "inspect": ("vector", "prompt", "meter", "mask"),
    "grade": ("vector", "prompt", "mask", "strengths"),
    "generate-pairs": ("concept", "llm_provider", "llm_model", "name"),
    "rebuild": ("target", "families", "pooling"),
    "from-md": ("md_path", "name", "out"),
    "build-j-lens": ("verify_only",),
}

_METAVAR_CHOICES_RE = re.compile(r"^\{([^{}]+)\}$")


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, tuple[argparse.ArgumentParser, list[str]]]:
    """`{canonical name: (subparser, aliases)}` in the parser's own order."""
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    out: dict[str, tuple[argparse.ArgumentParser, list[str]]] = {}
    by_id: dict[int, str] = {}
    for name, sp in sub.choices.items():
        if id(sp) in by_id:
            out[by_id[id(sp)]][1].append(name)
            continue
        by_id[id(sp)] = name
        out[name] = (sp, [])
    return out


def _method_help(parser: argparse.ArgumentParser) -> dict[str, str]:
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return {ca.dest.split(" ")[0]: str(ca.help or "") for ca in sub._choices_actions}


def _visible_actions(sp: argparse.ArgumentParser) -> list[argparse.Action]:
    return [
        a
        for a in sp._actions
        if not isinstance(a, argparse._HelpAction) and a.help is not argparse.SUPPRESS
    ]


def _describe(sp: argparse.ArgumentParser, method: str) -> list[dict[str, Any]]:
    """One entry per destination the page can set, in the parser's order."""
    primary = PRIMARY.get(method)
    by_dest: dict[str, list[argparse.Action]] = {}
    for a in _visible_actions(sp):
        by_dest.setdefault(a.dest, []).append(a)
    out: list[dict[str, Any]] = []
    for dest, acts in by_dest.items():
        if dest in SERVER_SET.get(method, {}):
            continue
        first = acts[0]
        positional = not first.option_strings
        entry: dict[str, Any] = {
            "dest": dest,
            "flags": [f for a in acts for f in a.option_strings] or [dest],
            "positional": positional,
            "help": " / ".join(dict.fromkeys(str(a.help) for a in acts if a.help)),
            "required": bool(getattr(first, "required", False)) or (positional and first.nargs is None),
            "default": sp.get_default(dest),
            "choices": None,
            "suggest": None,
            "widget": WIDGETS.get(dest),
            "fixed": False,
        }
        trues = [a for a in acts if isinstance(a, argparse._StoreTrueAction)]
        falses = [a for a in acts if isinstance(a, argparse._StoreFalseAction)]
        if trues or falses:
            entry["kind"] = "bool"
            entry["default"] = bool(entry["default"])
            # A lone store_true whose default is already True cannot be turned off on the
            # command line either: the page shows it ticked and locked.
            entry["fixed"] = bool(trues and not falses and entry["default"])
        elif isinstance(first, argparse._AppendAction) or first.nargs in ("+", "*"):
            entry["kind"] = "list"
            entry["default"] = list(entry["default"] or [])
        elif first.choices:
            entry["kind"] = "choice"
            entry["choices"] = [str(c) for c in first.choices]
        elif first.type in (int, float):
            entry["kind"] = "int" if first.type is int else "float"
        else:
            entry["kind"] = "text"
            m = _METAVAR_CHOICES_RE.match(str(first.metavar or ""))
            if m:
                # The parser's own statement of the usual values (`{both,fl2va,ref2va}`); it also
                # takes others (family aliases), so the page suggests and does not restrict.
                entry["suggest"] = [c.strip() for c in m.group(1).split(",")]
        if primary is not None:
            entry["primary"] = dest in primary
        else:
            entry["primary"] = bool(entry["required"] or positional)
        out.append(entry)
    if primary is not None:
        order = {d: i for i, d in enumerate(primary)}
        out.sort(key=lambda e: (0, order[e["dest"]]) if e["dest"] in order else (1, 0))
    return out


def method_schema() -> dict[str, Any]:
    """What `GET /api/schema` returns: the CLI's methods, and the node's lists and defaults."""
    parser = build_parser()
    helps = _method_help(parser)
    methods: dict[str, Any] = {}
    for name, (sp, aliases) in _subparsers(parser).items():
        if name in SKIP_METHODS:
            continue
        methods[name] = {
            "name": name,
            "aliases": aliases,
            "help": helps.get(name, ""),
            "server_set": sorted(SERVER_SET.get(name, {})),
            "args": _describe(sp, name),
        }
    return {
        "methods": methods,
        "node": {
            "apply_modes": list(cv.APPLY_MODES),
            "mask_modes": list(h3_spans.INFERENCE_MASK_MODES),
            "prompt_fields": list(h3_spans.PROMPT_FIELDS),
            "fields_join": h3_spans.FIELDS_JOIN,
            "defaults": dict(h3_rules.CONTROL_VECTOR_DEFAULTS),
        },
    }


def argv_for(method: str, values: dict[str, Any]) -> list[str]:
    """The argument list the command line would carry for `values` (`{dest: value}`).

    An empty or missing value is left off, so the parser's default stands. Optionals are written
    `--flag=value` and positionals follow a `--`, so a value that starts with a dash (a negative
    strength, a prompt) is never read as a flag. Raises `ValueError` on a name the method does not
    have, or one the studio sets itself."""
    subs = _subparsers(build_parser())
    if method in SKIP_METHODS or method not in subs:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(set(subs) - SKIP_METHODS)}")
    sp, _ = subs[method]
    if not isinstance(values, dict):
        raise ValueError("'args' must be an object of {argument: value}")
    by_dest: dict[str, list[argparse.Action]] = {}
    for a in _visible_actions(sp):
        by_dest.setdefault(a.dest, []).append(a)
    server_set = SERVER_SET.get(method, {})
    unknown = sorted(k for k in values if k not in by_dest)
    if unknown:
        raise ValueError(f"{method} has no argument(s) {unknown}")
    refused = sorted(k for k in values if k in server_set)
    if refused:
        raise ValueError(f"{method}: {refused} are set by the studio and cannot be sent")

    optionals: list[str] = []
    positionals: list[str] = []
    for dest, acts in by_dest.items():
        if dest not in values:
            continue
        val = values[dest]
        first = acts[0]
        trues = [a for a in acts if isinstance(a, argparse._StoreTrueAction)]
        falses = [a for a in acts if isinstance(a, argparse._StoreFalseAction)]
        if trues or falses:
            if not isinstance(val, bool):
                raise ValueError(f"{method}: {dest} takes true or false, got {val!r}")
            if val and trues:
                optionals.append(trues[0].option_strings[0])
            elif not val and falses:
                optionals.append(falses[0].option_strings[0])
            continue
        if isinstance(first, argparse._AppendAction) or first.nargs in ("+", "*"):
            items = [str(v) for v in (val if isinstance(val, (list, tuple)) else [val]) if str(v).strip()]
            if not first.option_strings:
                positionals.extend(items)
            else:
                optionals.extend(f"{first.option_strings[0]}={item}" for item in items)
            continue
        if val is None or (isinstance(val, str) and not val.strip()):
            continue
        if isinstance(val, bool):
            raise ValueError(f"{method}: {dest} takes a value, got {val!r}")
        if not first.option_strings:
            positionals.append(str(val))
        else:
            optionals.append(f"{first.option_strings[0]}={val}")
    return [method, *optionals, *(["--", *positionals] if positionals else [])]


def parse_request(method: str, values: dict[str, Any]) -> argparse.Namespace:
    """Parse `values` with the CLI's parser, as `argv_for` writes them. A parser refusal comes
    back as `ValueError` carrying the parser's own message."""
    argv = argv_for(method, values)
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            return build_parser().parse_args(argv)
    except SystemExit:
        lines = [ln.strip() for ln in err.getvalue().splitlines() if ln.strip()]
        msg = lines[-1] if lines else "the parser refused the arguments"
        raise ValueError(re.sub(r"^.*?: error: ", "", msg)) from None
