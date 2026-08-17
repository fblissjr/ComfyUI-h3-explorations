#!/usr/bin/env python3
"""Check saved UI workflows against a live ComfyUI's /object_info.

`build_workflows.py` already validates the API graphs it emits. That check
cannot see this class of bug at all: an API graph is `{node_id: {class_type,
inputs}}` with no widget list and no slot table, so widget/socket confusion,
widget ordering, and link-table corruption are all invisible there. Those
live only in the UI graph -- the file you actually open in ComfyUI.

The bug that motivated this: Sol-Attn added `tau_profile` as a
`force_input=True` input, which makes it a socket, not a widget. The
generator emitted it as a 13th widget value on a node with 12 widgets. Every
API graph validated clean through all of it.

What is checked, and what breaks if a check is deleted:

  type registered      a node type no longer installed renders as a red box
                       and the graph cannot run
  socket exists/typed  a slot the node does not have is a dangling wire; a
                       mistyped one connects things ComfyUI would refuse
  required connected   an unconnected required socket fails at queue time,
                       after the model has already loaded
  widget count         the count this file exists for. Values map to widgets
                       positionally, so a miscount silently shifts every
                       later widget onto the wrong control
  combo membership     a stale combo value (renamed sampler, deleted model
                       file) falls back to some other entry without saying so
  widget types         a string where a float belongs
  link integrity       ids unique, endpoints present, slots in range, and
                       both endpoints agree with the link table. LiteGraph
                       silently drops links it cannot resolve, which turns
                       into "why is this node not receiving anything"
  last_*_id            too low and the editor reissues a live id on the next
                       added node, splicing it onto an existing wire

Calibration, because a checker nobody has seen fail is decoration: it must
report 0 problems on a graph ComfyUI itself wrote. Round-trip any workflow
through the editor (open it, save it) and run this against the saved copy.
Two false-positive classes were found exactly that way and are suppressed
below -- widget-backed input slots, and dynamic slots.

    python bench/check_workflow_schema.py workflows/*.json
    python bench/check_workflow_schema.py --url http://127.0.0.1:8188 wf.json

Exit codes, and the distinction is the point:

    0   UI graphs were checked and they agree with the server's schema
    1   a graph DISAGREES -- a real finding, the thing this exists to catch
    2   this check DID NOT RUN, and nothing was validated

Two conditions produce 2: no reachable `/object_info` (no `--object-info`, no
live ComfyUI), and no UI graph among the paths given. Both used to be
indistinguishable from a verdict -- the first exited 1 like a genuine schema
violation, the second exited 0 after validating nothing. A check that says
"pass" when it looked at nothing, or "fail" when it could not look, teaches you
to disbelieve it either way. Same convention as `check_single_frame.py` and
`check_distill_settings.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

# Node types with no schema to check them against.
NON_NODES = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode"}

# Widgets the frontend adds that no schema declares, by node type. Each entry
# is how many extra trailing widget values that node legitimately carries.
# LoadImage's is the "choose file to upload" button.
EXTRA_WIDGETS = {"LoadImage": 1, "LoadImageMask": 1, "LoadAudio": 1, "LoadVideo": 1}


# Inputs whose node validates against the filesystem instead of its own combo
# list, so a value the combo does not offer can still be legal.
#
# `LoadImage` builds its combo from a NON-RECURSIVE `os.listdir` of the input
# directory (`ComfyUI/nodes.py::LoadImage.INPUT_TYPES`), so nothing in a
# subfolder ever reaches `/object_info`. It also defines `VALIDATE_INPUTS` ->
# `folder_paths.exists_annotated_filepath`, and ComfyUI's executor skips its
# own combo check for any input the node validates itself. So
# `h3_refs/face_x.png` submits and renders, and flagging it here would be this
# checker being stricter than the server -- the same class of defect as being
# looser, which cost a day on 2026-08-13 in the other direction.
#
# Read out of both files on 2026-08-16 rather than inferred from a render.
#
# **The UI cost, which is real and is not a defect:** the frontend populates
# that dropdown from the same list, so opening one of these graphs shows the
# subfolder value in the widget but will not offer it in the menu. It renders
# correctly; re-picking it from the dropdown is what you cannot do.
#
# Bare filenames are still checked against the list. Only values carrying a
# subfolder are exempt, which is exactly the case `/object_info` cannot see.
ANNOTATED_INPUTS = {("LoadImage", "image")}


def annotated_path(class_type, name, val):
    """True when this widget legally holds a path the combo list cannot show."""
    return ((class_type, name) in ANNOTATED_INPUTS
            and isinstance(val, str) and "/" in val)


def is_combo(t):
    # V1 serialized a combo as a bare list of choices; V3 emits "COMBO" with
    # the choices under opts["options"], and a dynamic combo as its own type.
    return isinstance(t, list) or t in ("COMBO", "COMFY_DYNAMICCOMBO_V3")


def choices_of(t, opts):
    if isinstance(t, list):
        return t
    got = opts.get("options")
    if not got:
        return None
    # dynamic combos list {"key": ..., "inputs": ...} rather than bare strings
    return [c["key"] if isinstance(c, dict) else c for c in got]


def _inputs(spec):
    for section in ("required", "optional"):
        for name, val in (spec.get("input", {}).get(section) or {}).items():
            opts = val[1] if len(val) > 1 and isinstance(val[1], dict) else {}
            yield section, name, val[0], opts


def widget_inputs(spec):
    """Inputs that own a saved widget value, in the order they are saved.

    An INT with control_after_generate owns TWO values: the number and the
    control mode ("randomize"/"fixed"/...). Counting it as one reported a
    false positive on every RandomNoise in the repo.
    """
    out = []
    for _section, name, t, opts in _inputs(spec):
        if is_combo(t):
            out.append((name, "COMBO", choices_of(t, opts)))
        elif t in ("INT", "FLOAT", "STRING", "BOOLEAN"):
            if opts.get("forceInput"):
                continue                       # force_input makes it a socket
            out.append((name, t, None))
            if opts.get("control_after_generate"):
                out.append((name + "/control", "CONTROL", None))
    return out


def expand_dynamic_combo(spec, wants, values):
    """Insert a DynamicCombo's selected option's widgets after the combo.

    A `DynamicCombo` declares each option's inputs nested under `options`, so
    `widget_inputs` sees only the combo itself and the count comes up short by
    however many the selection reveals. The frontend renders the revealed
    widgets immediately after the combo, and this repo's generator writes them
    there; if that assumption is ever wrong, a round-trip through the editor
    shows up here as a count or type mismatch, which is the intended failure.
    """
    out = []
    for i, (name, typ, choices) in enumerate(wants):
        out.append((name, typ, choices))
        if typ != "COMBO":
            continue
        opts = next((o for _s, n, _t, o in _inputs(spec) if n == name), {}) or {}
        options = opts.get("options")
        if not isinstance(options, list) or not options or i >= len(values):
            continue
        chosen = values[i]
        for option in options:
            if not isinstance(option, dict) or option.get("key") != chosen:
                continue
            inner = option.get("inputs") or {}
            for section in ("required", "optional"):
                for nm, val in (inner.get(section) or {}).items():
                    o = val[1] if len(val) > 1 and isinstance(val[1], dict) else {}
                    if is_combo(val[0]):
                        out.append((nm, "COMBO", choices_of(val[0], o)))
                    else:
                        out.append((nm, val[0], None))
    return out


def format_widgets(spec, chosen):
    """Widgets a combo value pulls in, e.g. VHS_VideoCombine's per-format set.

    /object_info carries these inside the combo's own metadata rather than as
    declared inputs, because which ones exist depends on the value selected.
    Returns them in `widget_inputs` shape so both can be checked alike.
    """
    if not chosen:
        return []
    for _section, _name, _t, opts in _inputs(spec):
        formats = (opts or {}).get("formats")
        if isinstance(formats, dict) and chosen in formats:
            out = []
            for w in formats[chosen]:
                if not (isinstance(w, list) and w and isinstance(w[0], str)):
                    continue
                if len(w) > 1 and isinstance(w[1], list):
                    out.append((w[0], "COMBO", w[1]))
                elif len(w) > 1 and isinstance(w[1], str):
                    out.append((w[0], w[1], None))
            return out
    return []


def socket_inputs(spec):
    """Inputs that take a wire, name -> (type, optional)."""
    out = {}
    for section, name, t, opts in _inputs(spec):
        if is_combo(t):
            continue
        if t in ("INT", "FLOAT", "STRING", "BOOLEAN") and not opts.get("forceInput"):
            continue
        out[name] = (t, section == "optional")
    return out


def check(wf, object_info):
    problems = []
    nodes = {n["id"]: n for n in wf["nodes"]}

    def flag(node, msg):
        problems.append(f"node {node['id']} ({node['type']}): {msg}")

    for n in wf["nodes"]:
        t = n["type"]
        if t in NON_NODES:
            continue
        if t not in object_info:
            problems.append(f"node {n['id']}: type {t!r} is NOT registered")
            continue
        spec = object_info[t]
        socks = socket_inputs(spec)

        for slot in n.get("inputs", []):
            # The frontend materializes every widget as an input slot too,
            # tagged with "widget". Those are not sockets; checking them
            # against the socket list reported 42 fake problems on a graph
            # the frontend itself had just written.
            if "widget" in slot:
                continue
            name = slot["name"]
            # Dynamic slots on multi-input nodes are named "group.member"
            # (MiniMaxH3ReferenceToVideo's ref_images.ref_image_0). Only the
            # group is in the schema; the members are created on demand.
            base = name.split(".", 1)[0]
            if base in socks and base != name:
                continue
            if name not in socks:
                flag(n, f"input slot {name!r} does not exist on this node "
                        f"(has: {sorted(socks)})")
            elif slot.get("type") != socks[name][0]:
                flag(n, f"input {name!r} typed {slot.get('type')!r}, "
                        f"node declares {socks[name][0]!r}")

        declared = {s["name"] for s in n.get("inputs", []) if "widget" not in s}
        for name, (typ, optional) in socks.items():
            if optional or name in declared:
                if name in declared and not optional:
                    slot = next(s for s in n["inputs"] if s["name"] == name)
                    if slot.get("link") is None:
                        flag(n, f"required input {name!r} is not connected")
                continue
            flag(n, f"required input {name!r} ({typ}) has no slot")

        outs = spec.get("output", [])
        for i, slot in enumerate(n.get("outputs", [])):
            if i >= len(outs):
                flag(n, f"output slot {i} is beyond the node's {len(outs)}")
            elif slot.get("type") != outs[i]:
                flag(n, f"output {i} typed {slot.get('type')!r}, "
                        f"node declares {outs[i]!r}")

        wants = widget_inputs(spec)
        has = n.get("widgets_values")
        if isinstance(has, list):
            wants = expand_dynamic_combo(spec, wants, has)
        allowed = len(wants) + EXTRA_WIDGETS.get(t, 0)
        if isinstance(has, list) and not len(wants) <= len(has) <= allowed:
            flag(n, f"{len(has)} widgets_values, node has {len(wants)} widgets "
                    f"{[w[0] for w in wants]}")
        elif isinstance(has, dict):
            # Keyed widgets_values. A node whose widget set depends on another
            # widget cannot use positions: VHS_VideoCombine appends the chosen
            # format's own widgets (pix_fmt, crf, save_metadata, ...) after
            # `format`, so the frontend writes an object instead of a list.
            # Checking only lists would have skipped this node in silence,
            # which reads identical to a clean pass.
            by_name = {w[0]: w for w in wants}
            for extra in format_widgets(spec, has.get("format")):
                by_name.setdefault(extra[0], extra)
            for key, got in has.items():
                if key == "videopreview":
                    continue          # frontend DOM widget, not an input
                if key not in by_name:
                    flag(n, f"widget {key!r} is not an input of this node, "
                            f"nor a widget of format {has.get('format')!r}")
                    continue
                _name, typ, choices = by_name[key]
                if (typ == "COMBO" and choices and got not in choices
                        and not annotated_path(t, key, got)):
                    flag(n, f"widget {key!r} = {got!r} is not one of {choices[:8]}")
                elif typ in ("INT", "FLOAT") and not isinstance(got, (int, float)):
                    flag(n, f"widget {key!r} = {got!r} is not numeric")
                elif typ == "BOOLEAN" and not isinstance(got, bool):
                    flag(n, f"widget {key!r} = {got!r} is not a boolean")
            for name, typ, _c in wants:
                if name not in has:
                    flag(n, f"widget {name!r} is missing from widgets_values")
        elif isinstance(has, list):
            for (name, typ, choices), got in zip(wants, has):
                if typ == "CONTROL":
                    continue
                if typ == "COMBO":
                    if (choices and got not in choices
                            and not annotated_path(t, name, got)):
                        shown = choices[:8]
                        flag(n, f"widget {name!r} = {got!r} is not one of "
                                f"{shown}{'...' if len(choices) > 8 else ''}")
                elif typ == "BOOLEAN" and not isinstance(got, bool):
                    flag(n, f"widget {name!r} should be a bool, got {got!r}")
                elif typ in ("INT", "FLOAT") and not isinstance(got, (int, float)):
                    flag(n, f"widget {name!r} should be numeric, got {got!r}")

    seen = {}
    for link in wf.get("links", []):
        lid, src, sslot, dst, dslot = link[:5]
        if lid in seen:
            problems.append(f"link {lid}: duplicate id")
        seen[lid] = link
        if src not in nodes:
            problems.append(f"link {lid}: source node {src} is missing")
        if dst not in nodes:
            problems.append(f"link {lid}: target node {dst} is missing")
            continue
        d = nodes[dst]
        if dslot >= len(d.get("inputs", [])):
            problems.append(f"link {lid}: target node {dst} ({d['type']}) has "
                            f"{len(d.get('inputs', []))} input slots, "
                            f"link wants slot {dslot}")
        elif d["inputs"][dslot].get("link") != lid:
            problems.append(
                f"link {lid}: node {dst} ({d['type']}) slot {dslot} "
                f"({d['inputs'][dslot]['name']}) records link "
                f"{d['inputs'][dslot].get('link')!r}, link table says {lid}")
        if src in nodes:
            s = nodes[src]
            if sslot >= len(s.get("outputs", [])):
                problems.append(f"link {lid}: source node {src} ({s['type']}) "
                                f"has no output slot {sslot}")
            elif lid not in (s["outputs"][sslot].get("links") or []):
                problems.append(f"link {lid}: node {src} ({s['type']}) "
                                f"output {sslot} does not list it")

    for n in wf["nodes"]:
        for slot in n.get("inputs", []):
            if slot.get("link") is not None and slot["link"] not in seen:
                problems.append(
                    f"node {n['id']} ({n['type']}) input {slot['name']!r} "
                    f"references link {slot['link']}, which is not in the link table")
        for slot in n.get("outputs", []):
            for lid in (slot.get("links") or []):
                if lid not in seen:
                    problems.append(
                        f"node {n['id']} ({n['type']}) output {slot['name']!r} "
                        f"references link {lid}, which is not in the link table")

    if wf["nodes"]:
        top = max(n["id"] for n in wf["nodes"])
        if wf.get("last_node_id", 0) < top:
            problems.append(f"last_node_id {wf.get('last_node_id')} is below the "
                            f"highest node id {top}")
    top_link = max([l[0] for l in wf.get("links", [])] or [0])
    if wf.get("last_link_id", 0) < top_link:
        problems.append(f"last_link_id {wf.get('last_link_id')} is below the "
                        f"highest link id {top_link}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("workflows", nargs="+", help="UI-format workflow JSON")
    ap.add_argument("--url", default="http://127.0.0.1:8188",
                    help="running ComfyUI (default %(default)s)")
    ap.add_argument("--object-info", help="cached /object_info JSON instead of --url")
    args = ap.parse_args()

    if args.object_info:
        object_info = json.load(open(args.object_info))
    else:
        try:
            with urllib.request.urlopen(f"{args.url}/object_info", timeout=30) as r:
                object_info = json.load(r)
        except Exception as exc:
            print(f"SKIP  no /object_info at {args.url} "
                  f"({type(exc).__name__}: {exc})")
            print("Exit 2, not 1: this check DID NOT RUN. It needs a live ComfyUI, "
                  "or --object-info\nwith a saved copy. Nothing was validated, which "
                  "is not the same as nothing being wrong.")
            return 2

    failed = 0
    checked = 0
    skipped_api = 0
    for path in args.workflows:
        wf = json.load(open(path))
        if "nodes" not in wf:
            skipped_api += 1
            print(f"{path}: API format (no node list) -- skipped, this checks UI graphs")
            continue
        checked += 1
        problems = check(wf, object_info)
        if problems:
            failed += 1
            print(f"{path}: {len(problems)} problem(s)")
            for p in problems:
                print("  -", p)
        else:
            print(f"{path}: ok")

    # A run that validated nothing is not a pass. Every argument being an API
    # graph, or a glob matching none, both land here -- and both used to print
    # a tidy list of "skipped" lines and exit 0, which reads as green. This is
    # the emptiest-input case CLAUDE.md says to ask about: what would the input
    # have to look like for this to fail? Previously, nothing.
    if checked == 0:
        print(f"SKIP  no UI graph was checked ({skipped_api} API graph(s) skipped, "
              f"{len(args.workflows)} path(s) given)")
        print("Exit 2, not 0: this check DID NOT RUN. It reads UI graphs -- the ones "
              "with a\n`nodes` list -- and none were passed. Green here would mean "
              "'validated nothing'.")
        return 2

    if failed:
        print(f"FAIL  {failed} of {checked} UI graph(s) disagree with the server's "
              f"schema")
        return 1
    print(f"ok    {checked} UI graph(s) match /object_info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
