#!/usr/bin/env python3
"""The standalone `H3ExactLoRA` node computes what this pack's two nodes compute.

`standalone/h3_mutant_distill/` is published as its own repo
(`h3-mutant-distill`) with one node, `H3ExactLoRA`, written small instead of
bundling `pdd_lora.py` and its helpers. A rewrite is a new implementation,
so it is held to this pack's `MiniMaxH3PDDLoRA` (at `backbone_apply="exact
branch"`, heads patched) and `MiniMaxH3LoRABranch`, which rendered the recipes
the owner judged.

Two stages:

    static    CPU, no server. On the real files: the PDD backbone's LoRA
              branches are torch.equal to `lora_branch.parse_lora`'s;
              the head block each step selects matches `pdd_math.schedule_knots`
              on every recipe schedule; each fused head is torch.equal to
              `pdd_lora._FusedHeads`'.
    graphs    Writes each recipe's pack graph and its twin with the pack's
              nodes swapped for `H3ExactLoRA`, both saving latents, for
              `bench/run_graph_arms.py`. `compare` then asserts the twins'
              saved latents are torch.equal.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_mutant_parity.py [static]
    <comfy venv python> bench/check_mutant_parity.py graphs --out <dir>
    <comfy venv python> bench/check_mutant_parity.py compare <pack prefix> <ours prefix>

What `static` does NOT establish: the patch plumbing on a live model (object
patch keys, the wrapper reaching `transformer_options`, the adaln patches
matching). The render stage does.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent
sys.path.insert(0, str(COMFY))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "standalone"))
sys.path.insert(0, str(REPO / "workflows"))

import torch  # noqa: E402

import h3_config as C  # noqa: E402

LORAS = COMFY / "models" / "loras"
UNETS = COMFY / "models" / "diffusion_models"

#: The recipes, as the pack's graphs that the owner judged or that the
#: package ships. Each is swapped node for node in `graphs`.
#: The FlashGen recipes (t2v and ref2va finishes, late blocks, i2v and r2v
#: alone) and the FlashGen branch comparisons in `static` left with the
#: FlashGen and FastH3 lane on 2026-11-04 (`docs/roadmap.md`, "Closed lanes");
#: their pack graphs are gone. The package's published examples are its own.
RECIPES = {
    "pdd6": "workflows/h3_text_to_video_pdd_manual_sigmas_api.json",
    "i2v_pdd8": "workflows/distill_experiments/h3_first_frame_to_video_pdd_savelat_api.json",
    "r2v_pdd8": "workflows/h3_image_ref_plus_text_to_video_pdd_api.json",
}

#: The files `static` compares, per checkpoint partition: (PDD sidecar,
#: checkpoint).
PARTITIONS = {
    "fl2va": (C.PDD_FL2VA_LORA, C.MODELS["unet_fl2va"]),
    "ref2va": (C.PDD_REF2VA_LORA, C.MODELS["unet_ref2va"]),
}

#: The sigma schedules whose PDD head blocks `static` compares.
SCHEDULES = {
    "pdd8": C.PDD8_SIGMAS,
    "pdd8_to_0.8 (a step switch's first pass)": "1.0, 0.988235, 0.972973, 0.952381, 0.923077, 0.878049, 0.8",
    "pdd6 (PDD_MANUAL_SIGMAS)": C.PDD_MANUAL_SIGMAS,
}


def _load(path):
    from safetensors.torch import load_file
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
    return load_file(str(path)), meta


def _branches_equal(ours, pack):
    if set(ours) != set(pack):
        return f"module sets differ: {len(ours)} against {len(pack)}"
    for k in ours:
        for slot in ("a", "b", "diff_b"):
            x, y = getattr(ours[k], slot), getattr(pack[k], slot)
            if (x is None) != (y is None) or (x is not None and not torch.equal(x, y)):
                return f"{k}.{slot} differs"
    return None


def static() -> int:
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import lora_branch as lb
    import pdd_lora as P
    import pdd_math as M
    from h3_mutant_distill import exact_lora as X

    fails = []

    def check(name, problem):
        print(f"  {'RED ' if problem else 'ok  '}  {name}{': ' + problem if problem else ''}")
        if problem:
            fails.append(name)

    from safetensors import safe_open
    for part, (pdd_file, unet) in PARTITIONS.items():
        pdd_sd, pdd_meta = _load(LORAS / pdd_file)

        dm_sd = {k: v for k, v in pdd_sd.items() if k.startswith("diffusion_model.")}
        check(f"{part}: PDD backbone branches",
              _branches_equal(X.parse_lora(dm_sd, 1.0), lb.parse_lora(dm_sd, 1.0)))

        shift_v, shift_a = float(pdd_meta["pdd_shift_video"]), float(pdd_meta["pdd_shift_audio"])
        num_steps = int(pdd_meta["pdd_num_steps"])
        with safe_open(str(UNETS / unet), "pt") as f:
            base = {st: (f.get_tensor(f"final_layer.{st}_out.weight").float(),
                         f.get_tensor(f"final_layer.{st}_out.bias").float()) for st in ("video", "audio")}
        bank = P._HeadBank({st: (pdd_sd[f"h3_pdd.bank.{st}.weight"].float(),
                                 pdd_sd[f"h3_pdd.bank.{st}.bias"].float(), *base[st])
                            for st in ("video", "audio")})

        for label, text in SCHEDULES.items():
            sched = torch.tensor([float(x) for x in text.split(",")], dtype=torch.float32)
            knots = M.schedule_knots(sched, shift_v, num_steps)
            want = list(zip(knots, knots[1:]))
            sc = X._Schedule(shift_v, num_steps)
            got = []
            for i in range(len(sched) - 1):
                sc.update({"sample_sigmas": sched, "sigmas": sched[i:i + 1]})
                got.append(sc.block)
            check(f"{part}: head blocks, {label}", None if got == want else f"{got} against {want}")
            for stream, shift in (("video", shift_v), ("audio", shift_a)):
                pack_h = P._FusedHeads(bank, stream, shift, num_steps, 1.0)
                ours_h = X._FusedHeads(pdd_sd[f"h3_pdd.bank.{stream}.weight"].float(),
                                       pdd_sd[f"h3_pdd.bank.{stream}.bias"].float(),
                                       *base[stream], shift, num_steps, 1.0)
                bad = [blk for blk in want
                       if not all(torch.equal(a, b) for a, b in
                                  zip(ours_h.get(blk, "cpu", torch.float32),
                                      pack_h.get(blk, "cpu", torch.float32)))]
                check(f"{part}: fused {stream} heads, {label}", f"blocks {bad} differ" if bad else None)

    print(f"\n{'RED' if fails else 'GREEN'}: {len(fails)} failure(s)")
    return 1 if fails else 0


def pin_pdd_sigmas(graph: dict) -> dict:
    """Feed the sampler a `ManualSigmas` where it read the PDD node's SIGMAS output.

    `H3ExactLoRA` has no SIGMAS output, and the node's own float32 schedule is
    not bit-identical to the 6-decimal string the examples use. Applied to
    BOTH arms of a pair, so they still differ only in the LoRA node."""
    g = copy.deepcopy(graph)
    nxt = max(int(k) for k in g) + 1
    for nid, n in list(g.items()):
        if n["class_type"] != "MiniMaxH3PDDLoRA" or \
                not any(v == [nid, 1] for m in g.values() for v in m["inputs"].values()):
            continue
        steps = n["inputs"].get("steps")
        if isinstance(steps, list):
            steps = g[str(steps[0])]["inputs"]["value"]
        if int(steps) != C.PDD_STEPS:
            raise ValueError(f"node {nid} emits {steps} steps; only PDD8's schedule is pinned")
        sig = str(nxt)
        nxt += 1
        g[sig] = {"class_type": "ManualSigmas", "inputs": {"sigmas": C.PDD8_SIGMAS}}
        for m in g.values():
            for k, v in m["inputs"].items():
                if v == [nid, 1]:
                    m["inputs"][k] = [sig, 0]
    return g


def swap(graph: dict) -> dict:
    """The pack graph with `MiniMaxH3PDDLoRA` and `MiniMaxH3LoRABranch` replaced by `H3ExactLoRA`."""
    g = copy.deepcopy(graph)
    for nid, n in g.items():
        ct, ins = n["class_type"], n["inputs"]
        if ct == "MiniMaxH3PDDLoRA":
            if ins.get("backbone_apply") != "exact branch" or not ins.get("patch_heads", True) \
                    or float(ins.get("head_strength", -1.0)) not in (-1.0, float(ins["strength"])):
                raise ValueError(f"node {nid}: a PDD setting H3ExactLoRA does not cover: {ins}")
            g[nid] = {"class_type": "H3ExactLoRA",
                      "inputs": {"model": ins["model"], "lora_name": ins["lora_name"],
                                 "strength": ins["strength"], "blocks": "all"}}
        elif ct == "MiniMaxH3LoRABranch":
            if ins.get("modules", "all") != "all" or \
                    (ins.get("start_percent", 0.0), ins.get("end_percent", 1.0)) != (0.0, 1.0):
                raise ValueError(f"node {nid}: a branch setting H3ExactLoRA does not cover: {ins}")
            g[nid] = {"class_type": "H3ExactLoRA",
                      "inputs": {"model": ins["model"], "lora_name": ins["lora_name"],
                                 "strength": ins["strength"], "blocks": ins.get("blocks", "all")}}
    # A consumer of the PDD node's SIGMAS output would lose its input.
    for nid, n in g.items():
        for v in n["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and g.get(str(v[0]), {}).get("class_type") \
                    == "H3ExactLoRA" and v[1] != 0:
                raise ValueError(f"node {nid} reads output {v[1]} of a swapped node")
    return g


def with_latents(graph: dict, prefix: str) -> dict:
    """Save the final latent, video and audio, of every `SamplerCustomAdvanced` nobody else samples from."""
    g = copy.deepcopy(graph)
    fed = {str(v[0]) for n in g.values() for v in n["inputs"].values()
           if isinstance(v, list) and len(v) == 2 and n["class_type"] == "SamplerCustomAdvanced"}
    finals = [nid for nid, n in g.items()
              if n["class_type"] == "SamplerCustomAdvanced" and nid not in fed]
    nxt = max(int(k) for k in g) + 1
    for nid in finals:
        # The AV latent is nested; split it as the pack's `_savelat` twins do.
        sep = str(nxt)
        g[sep] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": [nid, 0]}}
        for slot, stream in ((0, "video"), (1, "audio")):
            g[str(nxt + 1 + slot)] = {"class_type": "SaveLatent",
                                      "inputs": {"samples": [sep, slot],
                                                 "filename_prefix": f"latents/{prefix}_{stream}"}}
        nxt += 3
    return g


def graphs(out: Path, only=None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    for name, rel in RECIPES.items():
        if only and name not in only:
            continue
        pack = pin_pdd_sigmas(json.loads((REPO / rel).read_text()))
        for arm, g in (("pack", pack), ("ours", swap(pack))):
            path = out / f"parity_{name}_{arm}_api.json"
            path.write_text(json.dumps(with_latents(g, f"mutant_parity_{name}_{arm}"), indent=2) + "\n")
            print(f"  wrote {path}")
    return 0


def compare(a: Path, b: Path) -> int:
    from safetensors.torch import load_file      # a `.latent` is a safetensors file
    la, lb_ = load_file(str(a)), load_file(str(b))
    keys = sorted(set(la) | set(lb_))
    bad = [k for k in keys if k not in la or k not in lb_ or not torch.equal(la[k], lb_[k])]
    for k in keys:
        same = k in la and k in lb_ and torch.equal(la[k], lb_[k])
        print(f"  {'ok  ' if same else 'RED '}  {k}")
    print(f"\n{'RED' if bad else 'GREEN'}: {a.name} against {b.name}")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="H3ExactLoRA against this pack's PDD and branch nodes.")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("static")
    g = sub.add_parser("graphs")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--only", nargs="*", help="recipe names (default: all)")
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    args = ap.parse_args(argv)
    if args.cmd in (None, "static"):
        return static()
    if args.cmd == "graphs":
        return graphs(args.out, args.only)
    return compare(args.a, args.b)


if __name__ == "__main__":
    raise SystemExit(main())
