#!/usr/bin/env python3
"""What is INSIDE each shipped model file, asserted against a committed baseline.

## The escape this exists for

`check_model_files.py` asks `/object_info` whether a NAME resolves for the node
class that loads it. It never opens a file. So every name check stays green
while the bytes behind a name change: a re-download, a reship of the same
filename by the publisher, a local rebuild, a conversion between the two on-disk
quantization mechanisms. `convert_old_quants` makes that last one invisible at
load, because a file carrying only `__metadata__["_quantization_metadata"]` and
a file carrying per-layer `.comfy_quant` tensors behave identically once loaded
and have completely different headers.

Nothing else looks either. `analyze_quant_delta.py` reads two files deeply, by
name, for one comparison. This reads every model the repo names.

## What it asserts

For each model in `h3_config.MODELS` plus `IMAGE_VAE`: the quantization
fingerprint recorded in `bench/results/model_contents_baseline.json` still
describes the file on disk. The fingerprint is the set of distinct configs, how
many layers hold each, which module roles those are, and how many 2-D weights
carry no config at all. Red means the file changed under a name that did not.

The baseline is committed and moves deliberately: `--update-baseline` rewrites
it and prints what moved. A fingerprint that changed because the owner replaced
a checkpoint is a baseline update; one that changed because a download was
silently reshipped is the thing this exists to catch. The script cannot tell
those apart and does not try -- it reports the difference and a person decides.

## Reading the configs: every blob, never a sample

Grouping `.comfy_quant` blobs by byte length and decoding one per length class
is NOT sound, and both counterexamples are real rather than constructed:

  - `convrot_groupsize` 64 and 16 are the same width, so one length class held
    two configs in Comfy-Org's Qwen-Image prompt-expander checkpoints, and a
    one-blob sample reported the wrong group size for 27 layers.
  - a shorter format name buys back exactly the bytes `full_precision_matrix_mult`
    costs: `{"format":"nvfp4","convrot":true,"full_precision_matrix_mult":true}`
    is the same 72 bytes as this repo's shipped `int8_tensorwise` g256 config,
    so a flagged layer can hide inside an unflagged class.

The blobs are tens of bytes each and tens of KB per file. Decode them all.
(Technique and the first counterexample: a peer session, 2026-09-20.)

## Matched-shape controls, and why `--report` prints their absence

`--report` prints the census: configs, module roles with `in=`/`out=`, the 2-D
weights carrying no config, and any shape whose layers get DIFFERENT treatment
inside one file. That last one is the only evidence here that separates role
from shape. Where it exists, treatment varies while shape is held constant, so
the rule is positional. Where it does not, every treatment class is uniquely
shape-identified and the file cannot answer a role question however many layers
it has -- which is why absence is printed rather than omitted. On 2026-09-20
three role stories were proposed and dissolved before that line existed, and
the two text encoders turned out to be the only files here that carry no
control at all.

## A file that sets its own attention (2026-10-09)

Since ComfyUI `b26625f2` a model file can carry, per attention module, a list of
attention methods core will pick from, kitchen's Sol-Attn among them, and since
`f49c531e` a LoRA can set or replace that list
(`comfy/ldm/modules/attention.py::ComfyAttention`, `comfy/lora.py::load_lora`).
Core consults the list only when no node has set an attention override
(`wrap_attn`), so a graph that wires any attention node is not steered by it.
A graph with none is, silently, and that is the stamped dense baseline every
comparison is made against.

So every `.safetensors` name `h3_config` or a generated graph carries is opened
and its header read for a key ending in `.config`, which is the whole of what
core's two loaders look for. A file that has one is red unless the baseline
names it under `attention_preference_approved` with a reason (the owner,
2026-10-09: fail unless the file was approved). `scan_can_fail` writes a file
with such a key on every run and is red if the scan misses it, so a green here
is a scan that can see one. A name no file on this box answers to is counted
and not failed: whether a named file exists is `check_model_files.py`'s question.

Run with the ComfyUI venv python. No network, no GPU.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import struct
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "workflows"))

from h3_producer_provenance import producer_provenance  # noqa: E402

BASELINE = _HERE / "results" / "model_contents_baseline.json"

#: Collapse `blocks.17.attn.qkv_proj` and `model.layers.3.mlp.down_proj` to one
#: role each. Reasoned: a fingerprint keyed on individual layer indices would go
#: red on any depth change, which `analyze_adaln_pruning.py` already grades.
ROLE = lambda name: re.sub(r"\.\d+\.", ".N.", "." + name).lstrip(".")


def header(path: Path) -> dict:
    """The safetensors header dict, read without mapping the tensor data."""
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length))


def configs(path: Path, head: dict) -> tuple[dict, str | None]:
    """{module: config} for whichever on-disk mechanism the file uses."""
    markers = [k for k in head if k.endswith(".comfy_quant")]
    if markers:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return {k[: -len(".comfy_quant")]:
                    json.loads(bytes(handle.get_tensor(k).tolist()).decode())
                    for k in markers}, "comfy_quant"
    meta = head.get("__metadata__", {})
    if "_quantization_metadata" in meta:
        return json.loads(meta["_quantization_metadata"])["layers"], "_quantization_metadata"
    return {}, None


def shape(head: dict, module: str) -> tuple[int, int] | None:
    """(in_features, out_features) of a module's 2-D weight, or None."""
    dims = head.get(module + ".weight", {}).get("shape")
    return (dims[1], dims[0]) if dims and len(dims) == 2 else None


def fingerprint(path: Path) -> dict:
    head = header(path)
    cfg, mechanism = configs(path, head)
    linears = {k[: -len(".weight")] for k in head
               if k.endswith(".weight") and len(head[k]["shape"]) == 2}
    classes: dict[str, list[str]] = collections.defaultdict(list)
    for module, conf in cfg.items():
        classes[json.dumps(conf, sort_keys=True)].append(module)
    return {
        "mechanism": mechanism,
        "tensors": len([k for k in head if k != "__metadata__"]),
        "quantized_layers": len(cfg),
        "configs": {
            conf: {
                "layers": len(mods),
                "roles": dict(sorted(collections.Counter(ROLE(m) for m in mods).items())),
            }
            for conf, mods in sorted(classes.items(), key=lambda kv: -len(kv[1]))
        },
        "unquantized_2d_weights": dict(sorted(
            collections.Counter(ROLE(k) for k in linears - set(cfg)).items())),
    }


def controls(path: Path) -> dict:
    """Shapes whose layers are treated differently inside one file."""
    head = header(path)
    cfg, _ = configs(path, head)
    linears = {k[: -len(".weight")] for k in head
               if k.endswith(".weight") and len(head[k]["shape"]) == 2}
    by_shape: dict = collections.defaultdict(lambda: collections.defaultdict(set))
    for module in linears:
        conf = cfg.get(module)
        key = json.dumps(conf, sort_keys=True) if conf else "NOT QUANTIZED"
        by_shape[shape(head, module)][key].add(ROLE(module))
    return {f"in={s[0]} out={s[1]}": {k: sorted(v) for k, v in t.items()}
            for s, t in sorted(by_shape.items()) if s and len(t) > 1}


def wanted(models_root: Path) -> dict[str, Path]:
    import h3_config
    names = dict(h3_config.MODELS)
    names["image_vae"] = h3_config.IMAGE_VAE
    found = {}
    for key, name in names.items():
        for sub in ("diffusion_models", "text_encoders", "vae"):
            candidate = models_root / sub / name
            if candidate.exists():
                found[key] = candidate
                break
        else:
            found[key] = models_root / "???" / name
    return found


#: What core's loaders read as an attention preference: the model loader a
#: `<module>.comfy_attention.config` tensor, the LoRA loader any key ending in
#: `.config`. Read from ComfyUI at 08ff3c11, 2026-10-09.
ATTENTION_SUFFIX = ".config"
#: The baseline's key for files the owner has accepted with a preference in them: {file name: reason}.
APPROVED = "attention_preference_approved"


def attention_preferences(path: Path) -> dict[str, list]:
    """{key: the methods its list names, in order} for every attention preference in a file. Empty for none."""
    keys = [k for k in header(path) if k != "__metadata__" and k.endswith(ATTENTION_SUFFIX)]
    if not keys:
        return {}
    from safetensors import safe_open
    found = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in keys:
            try:
                conf = json.loads(bytes(handle.get_tensor(key).tolist()).decode())
            except (ValueError, UnicodeDecodeError):
                conf = None             # a key core would try to read and this cannot: still reported
            entries = conf if isinstance(conf, list) else [conf]
            found[key] = [e.get("attention") if isinstance(e, dict) else None for e in entries]
    return found


def named_files(models_root: Path) -> tuple[dict[str, list[Path]], list[str]]:
    """({name: its files on disk}, names no file answers to) for every `.safetensors` name `h3_config`'s
    constants or a generated graph carries, bench graphs included. A name is matched by its file name
    anywhere under the models root, links followed, because a graph names a LoRA by a path under its own
    folder and a model by a bare name."""
    import h3_config
    names: set[str] = set()

    def walk(value) -> None:
        if isinstance(value, str):
            if value.endswith(".safetensors"):
                names.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                walk(item)

    for attr in dir(h3_config):
        if attr.isupper():
            walk(getattr(h3_config, attr))
    for graph in h3_config.graph_paths(_HERE.parent / "workflows", include_bench=True):
        walk(json.loads(Path(graph).read_text()))
    on_disk: dict[str, list[Path]] = collections.defaultdict(list)
    for root, _dirs, files in os.walk(models_root, followlinks=True):
        for name in files:
            if name.endswith(".safetensors"):
                on_disk[name].append(Path(root) / name)
    found, absent = {}, []
    for name in sorted(names):
        paths = on_disk.get(Path(name.replace("\\", "/")).name)
        if paths:
            found[name] = sorted(paths)
        else:
            absent.append(name)
    return found, absent


def scan_can_fail() -> bool:
    """Whether `attention_preferences` sees a preference written as core's own doc lays one out."""
    import torch
    from safetensors.torch import save_file
    key = "blocks.0.attn.comfy_attention.config"
    blob = json.dumps([{"attention": "comfy_kitchen_sol", "tau": 1.0}, {"attention": "comfy_kitchen_int8"}]).encode()
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "control.safetensors"
        save_file({key: torch.tensor(list(blob), dtype=torch.uint8), "blocks.0.attn.qkv.weight": torch.zeros(2, 2)},
                  str(path))
        return attention_preferences(path) == {key: ["comfy_kitchen_sol", "comfy_kitchen_int8"]}


def attention_scan(models_root: Path, approved: dict) -> tuple[list[str], str]:
    """(the lines that make the check red, one line saying what was read)."""
    red = []
    if not scan_can_fail():
        red.append("FAIL the attention scan did not see a preference in a file written to have one: "
                   "a green below would mean nothing")
    found, absent = named_files(models_root)
    carrying = 0
    for name, paths in found.items():
        for path in paths:
            prefs = attention_preferences(path)
            if not prefs:
                continue
            carrying += 1
            methods = sorted({str(m) for listed in prefs.values() for m in listed})
            if name in approved or path.name in approved:
                print(f"ok   {name}: sets its own attention on {len(prefs)} module(s) ({', '.join(methods)}); "
                      f"approved: {approved.get(name) or approved.get(path.name)}")
                continue
            red.append(f"FAIL {name}: sets its own attention on {len(prefs)} module(s) ({', '.join(methods)}). "
                       f"On a graph with no attention node core runs that and nothing shows it. Approve it "
                       f"under `{APPROVED}` in {BASELINE.name} with a reason, or do not load it.")
    stale = sorted(n for n in approved if n not in found and not any(p.name == n for ps in found.values() for p in ps))
    for name in stale:
        print(f"STALE {name}: approved in {BASELINE.name} and named by nothing now; remove the entry")
    summary = (f"{sum(len(p) for p in found.values())} file(s) for {len(found)} name(s) read for an attention "
               f"preference, {carrying} carry one; {len(absent)} name(s) not on disk and not read")
    return red, summary


def render(key: str, path: Path) -> None:
    fp = fingerprint(path)
    print(f"{key}: {path.name}  [{fp['mechanism'] or 'unquantized'}]  "
          f"{fp['tensors']} tensors, {fp['quantized_layers']} quantized, "
          f"{len(fp['configs'])} config(s)")
    head = header(path)
    cfg, _ = configs(path, head)
    for conf, detail in fp["configs"].items():
        flag = "   FLAGGED full_precision_matrix_mult" if json.loads(conf).get(
            "full_precision_matrix_mult") else ""
        print(f"   {detail['layers']:5d} x {conf}{flag}")
        for role, count in detail["roles"].items():
            example = next(m for m in cfg if ROLE(m) == role)
            dims = shape(head, example)
            print(f"           {count:5d} x {role:44s} "
                  + (f"in={dims[0]:<6d} out={dims[1]}" if dims else "in=?      out=?"))
    if fp["unquantized_2d_weights"]:
        print("   not quantized (2-D weights carrying no config):")
        for role, count in fp["unquantized_2d_weights"].items():
            print(f"           {count:5d} x {role}")
    found = controls(path)
    if found:
        print("   MATCHED-SHAPE CONTROLS (shape held constant, treatment varies -> positional):")
        for dims, treatments in found.items():
            print(f"       {dims}")
            for conf, roles in sorted(treatments.items()):
                print(f"           {conf}")
                print(f"               {', '.join(roles)}")
    else:
        print("   NO MATCHED-SHAPE CONTROLS: every treatment class here is uniquely")
        print("       shape-identified, so this file cannot separate role from shape")
        print("       for inclusion, flagging or group size. Read it against other")
        print("       files, never alone, for any role claim.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--models", type=Path, default=None,
                        help="models root (default: the ComfyUI checkout above this repo)")
    parser.add_argument("--report", action="store_true",
                        help="print the full census instead of asserting the baseline")
    parser.add_argument("--update-baseline", action="store_true",
                        help="rewrite the committed baseline and print what moved")
    parser.add_argument("--out", type=Path, default=None, help="write the report JSON here")
    args = parser.parse_args()

    models_root = args.models or (_HERE.parents[2] / "models")
    targets = wanted(models_root)

    if args.report:
        for key, path in targets.items():
            if path.exists():
                render(key, path)
            else:
                print(f"{key}: {path.name}  MISSING\n")
        found, absent = named_files(models_root)
        print("attention preferences (a key ending in `.config`), every file a constant or a graph names:")
        for name, paths in found.items():
            for path in paths:
                prefs = attention_preferences(path)
                print(f"   {name}: " + (f"{len(prefs)} module(s), "
                                        f"{sorted({str(m) for v in prefs.values() for m in v})}" if prefs else "none"))
        for name in absent:
            print(f"   {name}: not on disk")
        return 0

    current, missing = {}, []
    for key, path in targets.items():
        if not path.exists():
            missing.append(f"{key} ({path.name})")
            continue
        current[key] = fingerprint(path)

    stored = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    baseline = stored.get("models", {})
    approved = stored.get(APPROVED, {})
    moved =[k for k in sorted(set(baseline) & set(current)) if baseline[k] != current[k]]
    added = sorted(set(current) - set(baseline))
    # A model h3_config still names whose file is not on this box is MISSING,
    # not dropped. Until 2026-10-05 `dropped` was taken against what was read,
    # so one dangling symlink printed both "no longer named by h3_config" and
    # "named by h3_config, not on disk", and the first was false.
    dropped = sorted(set(baseline) - set(targets))
    absent = sorted((set(baseline) & set(targets)) - set(current))

    if args.update_baseline:
        # Its fingerprint is carried forward for the same reason: the baseline
        # cannot be regenerated without the file, and a file missing from one
        # box is not the owner replacing a checkpoint.
        for key in absent:
            current[key] = baseline[key]
            print(f"kept    {key}: not on disk, fingerprint carried from the "
                  f"previous baseline")
        BASELINE.write_text(json.dumps({
            "baseline": "quantization fingerprint of every model h3_config names",
            "asserted_by": Path(__file__).name,
            "models": current,
            # the owner's list, carried: a rewrite of the fingerprints is not a withdrawal of an approval
            APPROVED: approved,
            "producer": producer_provenance(__file__),
        }, indent=2) + "\n")
        for key in moved:
            print(f"moved   {key}")
        for key in added:
            print(f"added   {key}")
        for key in dropped:
            print(f"dropped {key}")
        print(f"wrote {BASELINE.name} for {len(current)} models")
        return 0

    if args.out:
        args.out.write_text(json.dumps(
            {"models": current, "producer": producer_provenance(__file__)}, indent=2) + "\n")

    if not baseline:
        print(f"FAIL no baseline at {BASELINE.name}; run --update-baseline once and commit it")
        return 1
    for key in moved:
        print(f"FAIL {key}: contents changed under an unchanged name")
        was, now = baseline[key], current[key]
        for field in ("mechanism", "tensors", "quantized_layers"):
            if was[field] != now[field]:
                print(f"       {field}: {was[field]} -> {now[field]}")
        if was["configs"] != now["configs"]:
            print(f"       configs: {sorted(was['configs'])} -> {sorted(now['configs'])}")
    for key in added:
        print(f"FAIL {key}: named by h3_config but absent from the baseline")
    for key in dropped:
        print(f"FAIL {key}: in the baseline but no longer named by h3_config")
    for name in missing:
        print(f"FAIL {name}: named by h3_config, not on disk")
    red, summary = attention_scan(models_root, approved)
    for line in red:
        print(line)
    ok = not (moved or added or dropped or missing or red)
    print(f"{len(current)}/{len(targets)} models read; {summary}; "
          f"{'ok' if ok else 'FAIL'} against {BASELINE.name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
