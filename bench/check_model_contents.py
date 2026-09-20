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

Run with the ComfyUI venv python. No network, no GPU.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import struct
import sys
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
        return 0

    current, missing = {}, []
    for key, path in targets.items():
        if not path.exists():
            missing.append(f"{key} ({path.name})")
            continue
        current[key] = fingerprint(path)

    baseline = json.loads(BASELINE.read_text())["models"] if BASELINE.exists() else {}
    moved = [k for k in sorted(set(baseline) & set(current)) if baseline[k] != current[k]]
    added = sorted(set(current) - set(baseline))
    dropped = sorted(set(baseline) - set(current))

    if args.update_baseline:
        BASELINE.write_text(json.dumps({
            "baseline": "quantization fingerprint of every model h3_config names",
            "asserted_by": Path(__file__).name,
            "models": current,
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
    ok = not (moved or added or dropped or missing)
    print(f"{len(current)}/{len(targets)} models read; "
          f"{'ok' if ok else 'FAIL'} against {BASELINE.name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
