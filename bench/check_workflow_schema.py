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

Exits non-zero if any file has problems, so it works in a pre-commit hook.
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
        allowed = len(wants) + EXTRA_WIDGETS.get(t, 0)
        has = n.get("widgets_values")
        if isinstance(has, list) and not len(wants) <= len(has) <= allowed:
            flag(n, f"{len(has)} widgets_values, node has {len(wants)} widgets "
                    f"{[w[0] for w in wants]}")
        elif isinstance(has, list):
            for (name, typ, choices), got in zip(wants, has):
                if typ == "CONTROL":
                    continue
                if typ == "COMBO":
                    if choices and got not in choices:
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
            sys.exit(f"could not read {args.url}/object_info ({exc}). Start ComfyUI, "
                     f"or pass --object-info with a saved copy.")

    failed = 0
    for path in args.workflows:
        wf = json.load(open(path))
        if "nodes" not in wf:
            print(f"{path}: API format (no node list) -- skipped, this checks UI graphs")
            continue
        problems = check(wf, object_info)
        if problems:
            failed += 1
            print(f"{path}: {len(problems)} problem(s)")
            for p in problems:
                print("  -", p)
        else:
            print(f"{path}: ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
