#!/usr/bin/env python3
"""`MiniMaxH3LoRABranch` computes the merged LoRA exactly, on every module an H3 LoRA targets.

`lora_branch.py` adds a LoRA at the call instead of merging it. The claim is
narrow: **on an unquantized model, the branch equals the merge.** So this runs
core's real `MiniMaxH3Model` (two blocks and a two-block token refiner, float32
on CPU) through the node's object patches, and compares its output with a copy
whose weights hold `W + strength * alpha / rank * B A` and `bias + diff_b`,
merged in float32. The synthetic LoRA targets every module kind the FlashGen
file does: `qkv_proj`, `out_proj`, `fc1`, `fc2`, the block and final-layer
`adaln_proj.linear` with a `diff_b`, and the token refiner's blocks.

**Three controls, each a plausible way to get it wrong that still produces
tensors of the right shape**, and each must move the output off the merge:

    unscaled   alpha / rank dropped (the stub's alpha is not its rank)
    no fc2     the MLP's forward patch left out, which is what patching
               `fc2.forward` would amount to on the int8 path
    no diff_b  the adaln bias delta dropped

Also: a key the node cannot place is refused, and a forward another node
already patches is refused.

**What this does NOT establish:** anything on the int8 checkpoint or the card.
That the branch keeps what the merge loses there is
`bench/results/2026-09-26_int8_lora_requant.json`'s measurement, and whether
it shows in a render is a render's.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_lora_branch.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent.parent))

import torch  # noqa: E402

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True
import comfy.model_patcher  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.ldm.minimax.model as mm_h3  # noqa: E402
import lora_branch as lb  # noqa: E402

HIDDEN, TEXT_DIM, TEXT_LEN, LATENT, AUDIO_T = 256, 64, 8, (2, 4, 4), 6
RANK, ALPHA, STRENGTH = 4, 6.0, 0.8
#: The ops class the int8 checkpoints load through, with no quant config, so
#: its Linear is the class the node meets on the card. `manual_cast` here let a
#: `torch.nn.Linear` isinstance test pass on CPU and refuse every real render
#: (2026-09-26).
OPS = comfy.ops.mixed_precision_ops({}, torch.float32)


class _Base(torch.nn.Module):
    def __init__(self, dm):
        super().__init__()
        self.diffusion_model = dm


def tiny():
    """Random weights on `manual_cast`, then loaded into an `OPS` model the way a
    checkpoint load fills it (its Linear creates `weight` at load)."""
    src = _build(comfy.ops.manual_cast)
    dm = _build(OPS, init=False)
    missing, unexpected = dm.load_state_dict(src.state_dict(), strict=False)
    assert not unexpected, unexpected
    dm.requires_grad_(False)
    return dm


def _build(ops, init=True):
    torch.manual_seed(0)
    dm = mm_h3.MiniMaxH3Model(
        hidden_size=HIDDEN, num_layers=2, token_refiner_num_layers=2, num_attention_heads=2,
        attention_head_dim=128, ffn_hidden_size=384, text_dim=TEXT_DIM, timestep_input_dim=32,
        time_embed_hidden_size=64, time_embed_dim=64, dtype=torch.float32, device="cpu",
        operations=ops)
    if not init:
        return dm
    with torch.no_grad():
        for name, p in dm.named_parameters():
            p.fill_(1.0) if "norm" in name else p.normal_(0.0, 0.05)
        n = dm.rope.inv_freq.numel()
        dm.rope.inv_freq.copy_(1.0 / (10000.0 ** (torch.arange(n, dtype=torch.float32) / n)))
    dm.requires_grad_(False)
    return dm


def targets(dm):
    out = []
    for stack in ("blocks", "token_refiner.blocks"):
        for i in range(2):
            for kind in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
                out.append(f"{stack}.{i}.{kind}")
    out += [f"blocks.{i}.adaln_proj.linear" for i in range(2)] + ["final_layer.adaln_proj.linear"]
    return out


def synthetic_lora(dm):
    g = torch.Generator().manual_seed(7)
    sd = {}
    for path in targets(dm):
        w = dm.get_submodule(path).weight
        k = f"diffusion_model.{path}"
        sd[f"{k}.lora_A.weight"] = torch.randn(RANK, w.shape[1], generator=g) * 0.1
        sd[f"{k}.lora_B.weight"] = torch.randn(w.shape[0], RANK, generator=g) * 0.1
        sd[f"{k}.alpha"] = torch.tensor(ALPHA)
        if "adaln" in path:
            sd[f"{k}.diff_b"] = torch.randn(w.shape[0], generator=g) * 0.1
    return sd


def merged(dm, sd):
    ref = copy.deepcopy(dm)
    with torch.no_grad():
        for path in targets(dm):
            mod = ref.get_submodule(path)
            k = f"diffusion_model.{path}"
            mod.weight += STRENGTH * ALPHA / RANK * (sd[f"{k}.lora_B.weight"] @ sd[f"{k}.lora_A.weight"])
            if f"{k}.diff_b" in sd:
                mod.bias += STRENGTH * sd[f"{k}.diff_b"]
    return ref


def run(dm):
    g = torch.Generator().manual_seed(1)
    video = torch.randn((1, 24) + LATENT, generator=g)
    audio = torch.randn((1, 32, 2, AUDIO_T), generator=g)
    context = torch.randn((1, TEXT_LEN, TEXT_DIM), generator=g)
    layout = mm_h3.PackedLayout(TEXT_LEN, *LATENT, AUDIO_T)
    with torch.no_grad():
        out = dm.forward([video, audio], torch.tensor([500.0]), context, transformer_options={},
                         minimax_payload={"layout": layout})
    return torch.cat([t.flatten() for t in out]).double()


def branched(dm, sd, drop=()):
    patcher = comfy.model_patcher.ModelPatcher(_Base(dm), load_device=torch.device("cpu"),
                                               offload_device=torch.device("cpu"))
    branches = lb.parse_lora({k: v for k, v in sd.items()
                              if not any(d in k for d in drop)}, STRENGTH)
    m = lb.attach(patcher, branches)
    saved = {}
    for key, fn in m.object_patches.items():
        path = key[len("diffusion_model."):]
        owner, attr = path.rsplit(".", 1)
        mod = dm.get_submodule(owner)
        saved[(owner, attr)] = mod.__dict__.get(attr)
        setattr(mod, attr, fn)
    try:
        return run(dm), m
    finally:
        for (owner, attr), old in saved.items():
            mod = dm.get_submodule(owner)
            if old is None:
                delattr(mod, attr)
            else:
                setattr(mod, attr, old)


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main() -> int:
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
        if not ok:
            fails.append(name)

    dm = tiny()
    sd = synthetic_lora(dm)
    base = run(dm)
    ref = run(merged(dm, sd))
    effect = rel(base, ref)
    out, m = branched(dm, sd)
    err = rel(out, ref)
    check("the LoRA moves the output", effect > 1e-2, f"{effect:.3g} from the base")
    check("the branch equals the float32 merge", err < 1e-5, f"relative {err:.3g}")
    n_patches = len(m.object_patches)
    check("fc2 goes through the MLP's forward, not fc2's",
          not any(k.endswith("fc2.forward") for k in m.object_patches)
          and sum(k.endswith("mlp.forward") for k in m.object_patches) == 4,
          f"{n_patches} object patches")

    # controls: each must depart from the merge by far more than the floor
    saved = lb.parse_lora

    def unscaled(sd_, strength):
        return saved({k: v for k, v in sd_.items() if not k.endswith(".alpha")}, strength)
    lb.parse_lora = unscaled
    try:
        bad, _ = branched(dm, sd)
    finally:
        lb.parse_lora = saved
    check("control: alpha dropped is caught", rel(bad, ref) > 100 * max(err, 1e-9),
          f"{rel(bad, ref):.3g}")
    bad, _ = branched(dm, sd, drop=("mlp.fc2",))
    check("control: fc2 left out is caught", rel(bad, ref) > 100 * max(err, 1e-9),
          f"{rel(bad, ref):.3g}")
    bad, _ = branched(dm, sd, drop=(".diff_b",))
    check("control: diff_b dropped is caught", rel(bad, ref) > 100 * max(err, 1e-9),
          f"{rel(bad, ref):.3g}")

    try:
        lb.parse_lora({"diffusion_model.blocks.0.attn.qkv_proj.hada_w1_a": torch.zeros(1)}, 1.0)
        check("an unplaceable key is refused", False)
    except ValueError:
        check("an unplaceable key is refused", True)
    patcher = comfy.model_patcher.ModelPatcher(_Base(dm), load_device=torch.device("cpu"),
                                               offload_device=torch.device("cpu"))
    patcher.add_object_patch("diffusion_model.blocks.0.attn.qkv_proj.forward", lambda x: x)
    try:
        lb.attach(patcher, lb.parse_lora(sd, 1.0))
        check("a forward another node patches is refused", False)
    except ValueError:
        check("a forward another node patches is refused", True)

    if fails:
        print(f"\n  FAIL  {len(fails)}: {fails}")
        return 1
    print("\n  ok    the branch is the merge on every module kind, and each control departs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
