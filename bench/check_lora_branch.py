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

**Key names** (2026-10-05). `native_keys` renames a Kohya-style file
(`lora_unet_<path with underscores>.lora_down.weight`) through core's own
table, `comfy.lora.model_lora_keys_unet`. The synthetic LoRA is re-spelled
that way and must still equal the merge; a Kohya name for a block the model
does not have must reach `parse_lora` unrenamed and be refused; a native file
must come back as the same object. The stub model here is not core's
`MiniMaxH3` class, so core's H3-only rule (the bare module path, no prefix) is
read in `comfy/lora.py` and not exercised.

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
import comfy.lora  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.ldm.minimax.model as mm_h3  # noqa: E402
import lora_branch as lb  # noqa: E402

HIDDEN, TEXT_DIM, TEXT_LEN, LATENT, AUDIO_T = 256, 64, 8, (2, 4, 4), 6
RANK, ALPHA, STRENGTH = 4, 6.0, 0.8
#: The ops class the int8 checkpoints load through, with no quant config, so
#: its Linear is the class the node meets on the card. `manual_cast` here let a
#: `torch.nn.Linear` isinstance test pass on CPU and refuse every real render
#: (2026-09-26).
OPS = comfy.ops.mixed_precision_ops({}, torch.float32)


class _Config:
    """What `comfy.lora.model_lora_keys_unet` reads off a model's config."""
    unet_config = {}


class _Base(torch.nn.Module):
    model_config = _Config()

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


def kohya_spelling(sd):
    """`sd` as a Kohya export names it: `lora_unet_<path with underscores>`,
    `lora_down` / `lora_up`. `diff_b` has no Kohya name and stays native, so
    the file is mixed, which the node must also take."""
    out = {}
    for key, t in sd.items():
        body = key[len("diffusion_model."):]
        for suf, kohya in ((".lora_A.weight", ".lora_down.weight"),
                           (".lora_B.weight", ".lora_up.weight"), (".alpha", ".alpha")):
            if body.endswith(suf):
                out["lora_unet_" + body[:-len(suf)].replace(".", "_") + kohya] = t
                break
        else:
            out[key] = t
    return out


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

    # the granular controls, each against a merge of exactly what it keeps
    def merged_subset(keep):
        ref_ = copy.deepcopy(dm)
        with torch.no_grad():
            for path in targets(dm):
                if not keep(path):
                    continue
                mod = ref_.get_submodule(path)
                k = f"diffusion_model.{path}"
                mod.weight += STRENGTH * ALPHA / RANK * (sd[f"{k}.lora_B.weight"] @ sd[f"{k}.lora_A.weight"])
                if f"{k}.diff_b" in sd:
                    mod.bias += STRENGTH * sd[f"{k}.diff_b"]
        return run(ref_)

    def branched_select(modules, blocks):
        saved = lb.parse_lora
        lb.parse_lora = lambda sd_, st: lb.select(saved(sd_, st), modules, lb.parse_blocks(blocks))
        try:
            return branched(dm, sd)[0]
        finally:
            lb.parse_lora = saved

    out_na = branched_select("no adaln", "all")
    ref_na = merged_subset(lambda p: "adaln_proj" not in p)
    check("modules 'no adaln' equals a merge without the adaln modules",
          rel(out_na, ref_na) < 1e-5 and rel(out_na, ref) > 1e-3,
          f"{rel(out_na, ref_na):.3g} from its merge, {rel(out_na, ref):.3g} from the full merge")
    out_b1 = branched_select("all", "1")
    ref_b1 = merged_subset(lambda p: p.startswith("blocks.1."))
    check("blocks '1' equals a merge of block 1 alone",
          rel(out_b1, ref_b1) < 1e-5 and rel(out_b1, ref) > 1e-3,
          f"{rel(out_b1, ref_b1):.3g}")
    out_ref_only = branched_select("all", "refiner")
    ref_ref_only = merged_subset(lambda p: p.startswith("token_refiner."))
    check("blocks 'refiner' equals a merge of token_refiner alone",
          rel(out_ref_only, ref_ref_only) < 1e-5 and rel(out_ref_only, ref) > 1e-3,
          f"{rel(out_ref_only, ref_ref_only):.3g}")
    out_ref_b1 = branched_select("all", "refiner,1,final_layer")
    ref_ref_b1 = merged_subset(lambda p: p.startswith(("token_refiner.", "blocks.1.", "final_layer.")))
    check("blocks 'refiner,1,final_layer' equals a merge of token_refiner, block 1, and final_layer",
          rel(out_ref_b1, ref_ref_b1) < 1e-5 and rel(out_ref_b1, ref) > 1e-3,
          f"{rel(out_ref_b1, ref_ref_b1):.3g}")
    w = lb._Window(0.5, 1.0)
    sched = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
    active = []
    for sig in sched[:-1]:
        w.update({"sample_sigmas": sched, "sigmas": sig.reshape(1)})
        active.append(w.active)
    check("the step window applies from its start fraction on",
          active == [False, False, True, True], str(active))
    gate = lb._Window(0.0, 1.0)
    gate.active = False
    saved = lb.parse_lora
    def gated(sd_, st):
        br = saved(sd_, st)
        for b in br.values():
            b.gate = gate
        return br
    lb.parse_lora = gated
    try:
        out_off = branched(dm, sd)[0]
    finally:
        lb.parse_lora = saved
    check("a closed window applies nothing", rel(out_off, base) < 1e-6,
          f"{rel(out_off, base):.3g} from the base")

    # fc2's branch runs a chunk of rows at a time. Its factors are fetched
    # once per call, not once per chunk, and a closed window writes out no
    # activation. Counted, because neither shows in the output.
    copies = {"n": 0}
    real_dev = lb._Branch.__dict__["_dev"]
    def counting(t, like):
        copies["n"] += 1
        return real_dev.__func__(t, like)
    acts = {"n": 0}
    real_act = comfy.ops.INPUT_ACT_EAGER["swiglu"]
    def spy(x):
        acts["n"] += 1
        return real_act(x)
    rows = lb.FC2_CHUNK_ROWS
    lb._Branch._dev = staticmethod(counting)
    comfy.ops.INPUT_ACT_EAGER["swiglu"] = spy
    try:
        branched(dm, sd)
        one_chunk, acts_one = copies["n"], acts["n"]
        lb.FC2_CHUNK_ROWS = 3
        copies["n"] = acts["n"] = 0
        out_chunked = branched(dm, sd)[0]
        many_chunks, acts_many = copies["n"], acts["n"]
        acts["n"] = 0
        run(dm)
        acts_base = acts["n"]
        acts["n"] = 0
        lb.parse_lora = gated
        try:
            branched(dm, sd)
        finally:
            lb.parse_lora = saved
        acts_closed = acts["n"]
    finally:
        lb.FC2_CHUNK_ROWS = rows
        lb._Branch._dev = real_dev
        comfy.ops.INPUT_ACT_EAGER["swiglu"] = real_act
    check("fc2's factors are copied once per call, however many chunks",
          many_chunks == one_chunk and acts_many > acts_one and rel(out_chunked, ref) < 1e-5,
          f"{one_chunk} copies at one chunk and {many_chunks} at several "
          f"({acts_one} against {acts_many} activation calls)")
    check("a closed window writes out no activation for fc2",
          acts_closed == acts_base and acts_one > acts_base,
          f"{acts_closed} activation calls closed, {acts_base} with no branch, {acts_one} open")

    try:
        lb.parse_lora({"diffusion_model.blocks.0.attn.qkv_proj.hada_w1_a": torch.zeros(1)}, 1.0)
        check("an unplaceable key is refused", False)
    except ValueError:
        check("an unplaceable key is refused", True)

    # Key names: a Kohya-style file, renamed through core's table.
    patcher = comfy.model_patcher.ModelPatcher(_Base(dm), load_device=torch.device("cpu"),
                                               offload_device=torch.device("cpu"))
    kohya = kohya_spelling(sd)
    n_kohya = sum(k.startswith("lora_unet_") for k in kohya)
    renamed = lb.native_keys(kohya, patcher)
    # The module path becomes native; the suffix keeps its Kohya spelling,
    # which `parse_lora` reads as it reads `lora_A` / `lora_B`.
    want = {k.replace(".lora_A.weight", ".lora_down.weight")
             .replace(".lora_B.weight", ".lora_up.weight"): v for k, v in sd.items()}
    check("a Kohya-style file is renamed to the native module paths",
          n_kohya > 0 and set(renamed) == set(want) and all(renamed[k] is want[k] for k in want),
          f"{n_kohya} Kohya-named keys of {len(kohya)}")
    out_kohya, _ = branched(dm, renamed)
    check("the renamed Kohya file equals the float32 merge", rel(out_kohya, ref) < 1e-5,
          f"relative {rel(out_kohya, ref):.3g}")
    check("a native file passes through untouched", lb.native_keys(sd, patcher) is sd)
    stray = dict(kohya)
    stray["lora_unet_blocks_9_attn_qkv_proj.lora_down.weight"] = torch.zeros(RANK, HIDDEN)
    left = lb.native_keys(stray, patcher)
    try:
        lb.parse_lora(left, 1.0)
        check("control: a Kohya name for a module the model lacks is refused", False)
    except ValueError as e:
        check("control: a Kohya name for a module the model lacks is refused",
              "lora_unet_blocks_9_attn_qkv_proj" in str(e), "left unrenamed, then refused by name")
    both = dict(kohya)
    both["diffusion_model.blocks.0.attn.qkv_proj.alpha"] = torch.tensor(ALPHA)
    try:
        lb.native_keys(both, patcher)
        check("control: two names for one tensor are refused", False)
    except ValueError:
        check("control: two names for one tensor are refused", True)
    patcher = comfy.model_patcher.ModelPatcher(_Base(dm), load_device=torch.device("cpu"),
                                               offload_device=torch.device("cpu"))
    patcher.add_object_patch("diffusion_model.blocks.0.attn.qkv_proj.forward", lambda x: x)
    try:
        lb.attach(patcher, lb.parse_lora(sd, 1.0))
        check("a forward another node patches is refused", False)
    except ValueError:
        check("a forward another node patches is refused", True)

    # A second model must not wrap the first's applied patches. 2026-09-26:
    # Turbo, then PDD, then FlashGen, each a clone of one loaded checkpoint,
    # rendered in one process. Each node read the module's CURRENT forward,
    # which was the previous model's still-applied branch, so FlashGen ran
    # base + Turbo + PDD + FlashGen. `patch_model` is emulated: set each object
    # patch on the model and record the original in the backup the clones share.
    patcher = comfy.model_patcher.ModelPatcher(_Base(dm), load_device=torch.device("cpu"),
                                               offload_device=torch.device("cpu"))
    other = {k: (v * 3 if k.endswith("lora_B.weight") else v) for k, v in sd.items()}
    first = lb.attach(patcher, lb.parse_lora(other, STRENGTH))
    for k, fn in first.object_patches.items():
        old = comfy.utils.set_attr(patcher.model, k, fn)
        first.object_patches_backup.setdefault(k, old)
    try:
        second = lb.attach(patcher, lb.parse_lora(sd, STRENGTH))
        for k, fn in second.object_patches.items():
            comfy.utils.set_attr(patcher.model, k, fn)
        stacked = run(dm)
    finally:
        for k, old in first.object_patches_backup.items():
            comfy.utils.set_attr(patcher.model, k, old)
        first.object_patches_backup.clear()
    check("a second model does not wrap the first's applied patches",
          rel(stacked, ref) < 1e-5, f"{rel(stacked, ref):.3g} from its own merge")

    # Core PR 16681 (open as of 2026-10-02) has `DiTBlock` hand the MLP its
    # residual add: `self.mlp(h, residual=x, gate=gate_mlp,
    # segments=mod_segments)` whenever the MLP carries no hook. An object
    # patch is not a hook, so the MLP patch gets those keywords. Emulated by
    # swapping that last line into core's own block forward for one run.
    stock = mm_h3.DiTBlock.forward

    def pr16681_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
        attention = self.attn if attention is None else attention
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = mm_h3._mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        x = mm_h3._mod_gate(x, gate_msa, attention(h, rope_freqs=rope_freqs,
                                                   transformer_options=transformer_options), mod_segments)
        h = mm_h3._mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return self.mlp(h, residual=x, gate=gate_mlp, segments=mod_segments)

    mm_h3.DiTBlock.forward = pr16681_forward
    try:
        out_pr, _ = branched(dm, sd)
    except TypeError as e:
        out_pr = None
        detail = f"the MLP patch refused the call: {e}"
    finally:
        mm_h3.DiTBlock.forward = stock
    if out_pr is not None:
        detail = f"{rel(out_pr, ref):.3g} from the merge"
    check("under PR 16681's MLP call the branch still equals the merge",
          out_pr is not None and rel(out_pr, ref) < 1e-5, detail)

    # Chaining multiple MiniMaxH3LoRABranch nodes:
    # 1. Two full LoRAs chained via attach(attach(patcher, b1), b2) equal the float32 merge of sd + sd2,
    #    and cloning m1 to m2 does not mutate m1's closures in place (Trap 1D).
    def run_patcher(m_obj):
        saved_p = {}
        for key, fn in m_obj.object_patches.items():
            path = key[len("diffusion_model."):]
            owner, attr = path.rsplit(".", 1)
            mod = dm.get_submodule(owner)
            saved_p[(owner, attr)] = mod.__dict__.get(attr)
            setattr(mod, attr, fn)
        try:
            return run(dm)
        finally:
            for (owner, attr), old in saved_p.items():
                mod = dm.get_submodule(owner)
                if old is None:
                    delattr(mod, attr)
                else:
                    setattr(mod, attr, old)

    sd2 = {k: (v * 0.5 if k.endswith("lora_B.weight") else (v * -0.4 if k.endswith(".diff_b") else v))
           for k, v in sd.items()}
    ref_two = run(merged(merged(dm, sd), sd2))
    ref_sd2 = run(merged(dm, sd2))
    patcher_chain = comfy.model_patcher.ModelPatcher(_Base(dm), load_device=torch.device("cpu"),
                                                     offload_device=torch.device("cpu"))
    m1 = lb.attach(patcher_chain, lb.parse_lora(sd, STRENGTH))
    m2 = lb.attach(m1, lb.parse_lora(sd2, STRENGTH))
    out_m2 = run_patcher(m2)
    out_m1_after = run_patcher(m1)
    check("chaining two LoRA branches equals the float32 merge of both",
          rel(out_m2, ref_two) < 1e-5 and rel(out_m2, ref) > 1e-3,
          f"{rel(out_m2, ref_two):.3g} from two-LoRA merge, {rel(out_m2, ref):.3g} from first")
    check("cloning and chaining a second LoRA does not mutate the first clone's closures",
          rel(out_m1_after, ref) < 1e-5,
          f"{rel(out_m1_after, ref):.3g} from first merge")

    # 2. Trap 1B: when m1 patches mlp.forward (because sd has mlp.fc2) and a second LoRA patches
    #    ONLY mlp.fc1 (no mlp.fc2 keys), install refreshes mlp.forward with the new fc1_forward.
    sd2_fc1_only = {k: v for k, v in sd2.items() if ".mlp.fc1." in k}
    ref_fc1_chain = copy.deepcopy(merged(dm, sd))
    with torch.no_grad():
        for path in targets(dm):
            if ".mlp.fc1" not in path:
                continue
            mod = ref_fc1_chain.get_submodule(path)
            k = f"diffusion_model.{path}"
            mod.weight += STRENGTH * ALPHA / RANK * (sd2_fc1_only[f"{k}.lora_B.weight"] @ sd2_fc1_only[f"{k}.lora_A.weight"])
    ref_fc1_out = run(ref_fc1_chain)
    m_fc1_chain = lb.attach(m1, lb.parse_lora(sd2_fc1_only, STRENGTH))
    out_fc1_chain = run_patcher(m_fc1_chain)
    check("chaining an mlp.fc1-only LoRA after a full LoRA refreshes mlp.forward",
          rel(out_fc1_chain, ref_fc1_out) < 1e-5 and rel(out_fc1_chain, ref) > 1e-4,
          f"{rel(out_fc1_chain, ref_fc1_out):.3g} from merge, {rel(out_fc1_chain, ref):.3g} from m1")

    # 3. Trap 1C: chained LoRAs with independent _Window gates filter each fc2 branch per step.
    b1_gated = lb.parse_lora(sd, STRENGTH)
    b2_gated = lb.parse_lora(sd2, STRENGTH)
    w1_gate = lb._Window(0.0, 0.5)
    w2_gate = lb._Window(0.5, 1.0)
    for br in b1_gated.values():
        br.gate = w1_gate
    for br in b2_gated.values():
        br.gate = w2_gate
    m_wchain = lb.attach(lb.attach(patcher_chain, b1_gated), b2_gated)
    w1_gate.active, w2_gate.active = True, False
    out_step_early = run_patcher(m_wchain)
    w1_gate.active, w2_gate.active = False, True
    out_step_late = run_patcher(m_wchain)
    check("chained LoRAs with independent step windows apply only the active branch at each step",
          rel(out_step_early, ref) < 1e-5 and rel(out_step_late, ref_sd2) < 1e-5,
          f"early={rel(out_step_early, ref):.3g}, late={rel(out_step_late, ref_sd2):.3g}")

    # 4. Bug A: MiniMaxH3LoRABranch.execute does not attach step _Window to token_refiner.* branches,
    #    and _window_wrapper resets window.active = True in finally.
    saved_get_path = lb.folder_paths.get_full_path_or_raise
    saved_load_tf = comfy.utils.load_torch_file
    lb.folder_paths.get_full_path_or_raise = lambda folder, name: f"/tmp/{name}"
    comfy.utils.load_torch_file = lambda path, safe_load=True: sd
    try:
        node_out = lb.MiniMaxH3LoRABranch.execute(
            patcher_chain, "synthetic_pika.safetensors", strength=STRENGTH,
            modules="all", blocks="all", start_percent=0.0, end_percent=0.5
        )[0]
    finally:
        lb.folder_paths.get_full_path_or_raise = saved_get_path
        comfy.utils.load_torch_file = saved_load_tf
    ref_fn = node_out.object_patches["diffusion_model.token_refiner.blocks.0.attn.qkv_proj.forward"]
    dit_fn = node_out.object_patches["diffusion_model.blocks.0.attn.qkv_proj.forward"]
    ref_br = ref_fn._h3_lora_linear[1][0]
    dit_br = dit_fn._h3_lora_linear[1][0]
    win_obj = dit_br.gate
    wrap_fn = lb._window_wrapper(win_obj)
    seen_inside = []
    wrap_fn(
        lambda x, t, c, transformer_options={}, **kw: seen_inside.append(win_obj.active),
        None, None, None,
        transformer_options={"sample_sigmas": sched, "sigmas": sched[3].reshape(1)},
    )
    check("a step window leaves token_refiner.* ungated and resets window.active after the step",
          ref_br.gate is None and win_obj is not None and seen_inside == [False] and win_obj.active is True,
          f"ref_gate={ref_br.gate}, seen_inside={seen_inside}, active_after={win_obj.active}")

    # 5. Unequal-rank 3-band block-diagonal qkv_proj (matching svd995 / svd99 (r_q, r_k, r_v)
    #    plus interleaved columns) compacts b from [3*D, R] to [D, R] and equals float32 merge,
    #    while dense qkv_proj stays unpacked (_b_splits is None).
    dense_qkv_br = lb.parse_lora(sd, STRENGTH)["blocks.0.attn.qkv_proj"]
    check("dense qkv_proj stays unpacked (_b_splits is None)",
          dense_qkv_br._b_splits is None and dense_qkv_br.b.shape == (3 * HIDDEN, RANK),
          f"shape={tuple(dense_qkv_br.b.shape)}, splits={dense_qkv_br._b_splits}")

    g_bd = torch.Generator().manual_seed(29)
    sd_bd = copy.deepcopy(sd)
    # Interleaved columns across bands (band 0: 3 cols, band 1: 1 col, band 2: 2 cols -> R=6)
    col_bands = (0, 2, 0, 1, 2, 0)
    r_bd = len(col_bands)
    for path in targets(dm):
        if not path.endswith("attn.qkv_proj"):
            continue
        k = f"diffusion_model.{path}"
        w = dm.get_submodule(path).weight
        d_band = w.shape[0] // 3
        a_bd = torch.randn(r_bd, w.shape[1], generator=g_bd) * 0.1
        b_bd = torch.zeros(w.shape[0], r_bd)
        for c_idx, b_idx in enumerate(col_bands):
            b_bd[b_idx * d_band : (b_idx + 1) * d_band, c_idx] = (
                torch.randn(d_band, generator=g_bd) * 0.1
            )
        sd_bd[f"{k}.lora_A.weight"] = a_bd
        sd_bd[f"{k}.lora_B.weight"] = b_bd
        sd_bd[f"{k}.alpha"] = torch.tensor(float(r_bd))

    ref_bd_dm = copy.deepcopy(dm)
    with torch.no_grad():
        for path in targets(dm):
            mod = ref_bd_dm.get_submodule(path)
            k = f"diffusion_model.{path}"
            r_cur = sd_bd[f"{k}.lora_A.weight"].shape[0]
            alpha_cur = float(sd_bd[f"{k}.alpha"])
            mod.weight += STRENGTH * alpha_cur / r_cur * (
                sd_bd[f"{k}.lora_B.weight"] @ sd_bd[f"{k}.lora_A.weight"]
            )
            if f"{k}.diff_b" in sd_bd:
                mod.bias += STRENGTH * sd_bd[f"{k}.diff_b"]
    ref_bd = run(ref_bd_dm)
    parsed_bd = lb.parse_lora(sd_bd, STRENGTH)
    qkv_bd_br = parsed_bd["blocks.0.attn.qkv_proj"]
    out_bd, _ = branched(dm, sd_bd)
    check("unequal-rank 3-band block-diagonal qkv_proj compacts b to [D, R] and equals float32 merge",
          qkv_bd_br._b_splits == (3, 1, 2)
          and qkv_bd_br.b.shape == (HIDDEN, r_bd)
          and rel(out_bd, ref_bd) < 1e-5,
          f"shape={tuple(qkv_bd_br.b.shape)}, splits={qkv_bd_br._b_splits}, rel={rel(out_bd, ref_bd):.3g}")

    # 6. LINEAR_CHUNK_ROWS multi-chunk linear execution copies factors once per module call
    #    and equals the float32 merge (including packed 3-band qkv_proj).
    lin_copies = {"n": 0}
    real_dev_lin = lb._Branch.__dict__["_dev"]

    def counting_lin(t, like):
        lin_copies["n"] += 1
        return real_dev_lin.__func__(t, like)

    saved_lin_chunk = lb.LINEAR_CHUNK_ROWS
    lb._Branch._dev = staticmethod(counting_lin)
    try:
        branched(dm, sd_bd)
        lin_one_chunk = lin_copies["n"]
        lb.LINEAR_CHUNK_ROWS = 3
        lin_copies["n"] = 0
        out_lin_chunked = branched(dm, sd_bd)[0]
        lin_many_chunks = lin_copies["n"]
    finally:
        lb.LINEAR_CHUNK_ROWS = saved_lin_chunk
        lb._Branch._dev = real_dev_lin
    check("linear branches copy factors once per module call across row chunks and equal merge",
          lin_many_chunks == lin_one_chunk and rel(out_lin_chunked, ref_bd) < 1e-5,
          f"{lin_one_chunk} copies at one chunk, {lin_many_chunks} at LINEAR_CHUNK_ROWS=3, "
          f"rel={rel(out_lin_chunked, ref_bd):.3g}")

    if fails:
        print(f"\n  FAIL  {len(fails)}: {fails}")
        return 1
    print("\n  ok    the branch is the merge on every module kind, and each control departs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
