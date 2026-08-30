"""Grade the un-merged backbone path against the merged one it replaces.

`unmerged_blocks` moves a block's backbone LoRA from a weight patch to a
forward patch. The claim that makes it worth having is narrow and checkable:
**the two paths compute the same thing.** So this asserts

    unmerged(x)  ==  x @ (W + (alpha/rank) * B @ A).T

on a stub Linear, in float64 -- no ComfyUI, no H3, no GPU. If that identity
does not hold, the knob is not an optimisation, it is a second model.

The identity is what makes the measured difference attributable. A merged
patch on an int8_convrot module is requantised with a recalculated scale
(`bench/results/2026-08-30_pdd_quant_interaction.json`), so IF the two paths
agree in exact arithmetic, the only thing left between them is that
quantisation. If they disagree here, that record measures nothing about this
knob.

**Four deliberate violations, run because the outcome was open rather than to
decorate a result.** Each is a way of getting the low-rank form subtly wrong
that still produces plausible tensors of the right shape:

    swapped        B @ A written as A @ B
    unscaled       alpha/rank dropped
    strengthless   `strength` dropped from the fold
    transposed     `x @ A.T` written the other way

`unscaled` is the one that earns the file. PDD's own alpha/rank is exactly
1.0 (alpha 64, rank 64), so on the shipped artifacts the correct and the
broken implementation agree to the bit -- a check built from the real file
alone would pass forever and fail on the first artifact with a different
alpha. The stub uses alpha != rank on purpose.

Also grades `split_unmerged` (which keys leave the weight-patch dict, and that
naming a module the file lacks refuses rather than silently merging), and
that Sol-Attn's composition filter does not claim these patch keys.

Exit 0 green, 1 red.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))          # ComfyUI root
sys.path.insert(0, str(HERE.parent))              # this repo

import pdd_lora as P                              # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def stub(out_f: int, in_f: int, rank: int, alpha: float, seed: int = 0):
    """A Linear and a LoRA pair with alpha != rank, in float64."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out_f, in_f, generator=g, dtype=torch.float64)
    a = torch.randn(rank, in_f, generator=g, dtype=torch.float64) * 0.05
    b = torch.randn(out_f, rank, generator=g, dtype=torch.float64) * 0.05
    return w, a, b, alpha


def merged(w, a, b, alpha, rank, strength):
    """What `comfy.lora` computes into the weight. The reference."""
    return w + strength * (alpha / rank) * (b @ a)


def run_unmerged(w, a, b, alpha, rank, strength, x, *, fold=None):
    """Drive the shipped forward factory exactly as the node drives it."""
    scaled = b * (strength * alpha / rank) if fold is None else fold
    fwd = P._make_unmerged_forward(lambda t: t @ w.T, a, scaled)
    return fwd(x)


def main() -> int:
    print("identity: un-merged forward against the merged weight")
    # alpha != rank deliberately: see the module docstring.
    out_f, in_f, rank, alpha = 96, 64, 8, 32.0
    w, a, b, alpha = stub(out_f, in_f, rank, alpha)
    x = torch.randn(5, in_f, generator=torch.Generator().manual_seed(7),
                    dtype=torch.float64)

    for strength in (1.0, 0.5, 0.0, -0.25, 2.0):
        ref = x @ merged(w, a, b, alpha, rank, strength).T
        got = run_unmerged(w, a, b, alpha, rank, strength, x)
        err = float((got - ref).abs().max())
        check(f"strength {strength:>5}", err < 1e-12, f"max abs {err:.2e}")

    print("\ndeliberate violations: each must MISS the merged reference")
    # SQUARE, so every transposition below is shape-VALID and has to be caught
    # by its numbers. On the rectangular stub above two of these raised a
    # shape error, which is a miss but a weaker one -- "it cannot run" is not
    # "it runs and is wrong", and only the second is what would ship.
    sq_f, sq_rank, sq_alpha = 48, 48, 24.0
    sw, sa, sb, sq_alpha = stub(sq_f, sq_f, sq_rank, sq_alpha, seed=3)
    sx = torch.randn(5, sq_f, generator=torch.Generator().manual_seed(11),
                     dtype=torch.float64)

    def sq_ref(strength):
        return sx @ merged(sw, sa, sb, sq_alpha, sq_rank, strength).T

    def sq_run(fold):
        return P._make_unmerged_forward(lambda t: t @ sw.T, sa, fold)(sx)

    scale = sq_alpha / sq_rank
    violations = {
        # each entry: (what the broken fold produces, the reference it must miss)
        "swapped B @ A -> A @ B": (
            lambda: P._make_unmerged_forward(
                lambda t: t @ sw.T, sb.T * scale, sa.T)(sx), sq_ref(1.0)),
        "unscaled (alpha/rank dropped)": (
            lambda: sq_run(sb), sq_ref(1.0)),
        # driven at 2.0 and graded against 2.0: dropping `strength` from the
        # fold must miss the arm it was asked for. The first version of this
        # case graded it against the 1.0 reference, which the broken fold
        # reproduces exactly -- the check went red on its own arithmetic
        # rather than on the code, which is the "red on correct state" shape.
        "strengthless (strength dropped)": (
            lambda: sq_run(sb * scale), sq_ref(2.0)),
        "transposed (x @ A instead of x @ A.T)": (
            lambda: P._make_unmerged_forward(
                lambda t: t @ sw.T, sa.T, sb * scale)(sx), sq_ref(1.0)),
    }
    for name, (fn, against) in violations.items():
        try:
            got = fn()
            missed = float((got - against).abs().max()) > 1e-6
            detail = "" if missed else "MATCHED the reference; blind to this"
        except Exception as exc:
            missed, detail = True, f"raised {type(exc).__name__} (weaker miss)"
        check(name, missed, detail)

    # And the control on the controls: the same square rig, driven CORRECTLY,
    # must match. Without this a violation could "miss" because the rig itself
    # is broken, and every row above would read as a pass.
    err = float((sq_run(sb * (2.0 * scale)) - sq_ref(2.0)).abs().max())
    check("square rig itself is correct at strength 2.0", err < 1e-12,
          f"max abs {err:.2e}")

    print("\nsigma gate: the window that makes a per-block arm controlled")

    class _Tk:
        def __init__(self, sigma=None): self.sigma = sigma

    w, a, b, alpha = stub(out_f, in_f, rank, alpha, seed=5)
    x2 = torch.randn(4, in_f, generator=torch.Generator().manual_seed(13),
                     dtype=torch.float64)
    scale = alpha / rank
    base_only = x2 @ w.T
    with_delta = x2 @ merged(w, a, b, alpha, rank, 1.0).T

    def gated(sigma, window):
        f = P._make_unmerged_forward(lambda t: t @ w.T, a, b * scale,
                                     tracker=_Tk(sigma), window=window)
        return f(x2)

    cases = [
        ("no window at all applies the delta", None, 0.5, with_delta),
        ("sigma inside the window applies it", (0.4, 0.9), 0.8, with_delta),
        ("sigma below the window does not", (0.4, 0.9), 0.2, base_only),
        ("sigma above the window does not", (0.4, 0.9), 0.95, base_only),
        ("sigma exactly at the low edge applies", (0.4, 0.9), 0.4, with_delta),
        ("sigma exactly at the high edge applies", (0.4, 0.9), 0.9, with_delta),
    ]
    for name, window, sigma, want in cases:
        got = gated(sigma, window)
        check(name, float((got - want).abs().max()) < 1e-12)

    # The one that matters, and it is a design choice rather than an accident:
    # a window whose tracker has never seen a sigma applies NOTHING. If it
    # defaulted open, a windowed arm on a graph where the capture patch never
    # ran would be silently identical to an unwindowed one -- the arms would
    # differ in the widget and not in the render, which is the worst shape a
    # control can have.
    got = gated(None, (0.4, 0.9))
    check("unset sigma with a window applies NOTHING",
          float((got - base_only).abs().max()) < 1e-12,
          "an unset gate must fail closed")
    got = gated(None, None)
    check("unset sigma with NO window still applies",
          float((got - with_delta).abs().max()) < 1e-12)

    print("\nsplit_unmerged: which keys leave the weight-patch dict")
    backbone = {}
    for i in (0, 7, 49):
        for kind in P.UNMERGED_KINDS:
            for slot in ("lora_A.weight", "lora_B.weight", "alpha"):
                backbone[f"diffusion_model.blocks.{i}.{kind}.{slot}"] = (
                    torch.tensor([[1.0]]) if slot != "alpha"
                    else torch.tensor(64.0))
    backbone["diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_A.weight"] = \
        torch.tensor([[1.0]])
    keep, lifted = P.split_unmerged(dict(backbone), frozenset({7}))
    n = len(P.UNMERGED_KINDS)
    check(f"lifts exactly the {n} un-mergeable modules of the named block",
          len(lifted) == n, f"got {len(lifted)}")
    check(f"lifts {3 * n} tensors out of the patch dict",
          len(keep) == len(backbone) - 3 * n, f"kept {len(keep)}")
    # The one that would have caught the shipped bug, and did not exist until
    # an observation capture found it by counting rows. `mlp.fc2`'s forward is
    # never called on the INT8 path -- `linear_input_act` reads the weight and
    # does the matmul itself -- so lifting it removes its LoRA from the weight
    # patch and applies nothing in its place.
    check("mlp.fc2 is NOT un-mergeable (its forward is never called)",
          "mlp.fc2" not in P.UNMERGED_KINDS,
          "lifting it silently drops the largest update in the file")
    _, lifted7 = P.split_unmerged(dict(backbone), frozenset({7}))
    check("and so it is never lifted",
          all(k != "mlp.fc2" for _, k in lifted7))
    check("leaves other blocks merged",
          all(".blocks.7." not in k for k in keep))
    check("never lifts the refiner",
          any("token_refiner" in k for k in keep))
    check("carries alpha and rank through",
          lifted[(7, "attn.qkv_proj")][2] == 64.0)

    print("\nsplit_unmerged: a module the file does not carry")
    try:
        P.split_unmerged(dict(backbone), frozenset({13}))
        check("refuses a block the file lacks", False, "returned quietly")
    except RuntimeError as exc:
        check("refuses a block the file lacks", "13" in str(exc))

    print("\nclash guard")
    check("clean model reports no clash",
          P.unmerged_patch_clash({}, frozenset({7})) == [])
    taken = P.unmerged_patch_clash(
        {P.unmerged_patch_key(7, "mlp.fc1"): object()}, frozenset({7}))
    check("a taken projection is reported", taken == [
        P.unmerged_patch_key(7, "mlp.fc1")], str(taken))

    print("\nSol-Attn composition filter must not claim these keys")
    # vendor/sol_attn_minimax.py: owner = key.rsplit(".", 2)[-2].lower(), and
    # it composes when "attn" is in that owner. If it claimed these, the LoRA
    # would apply only inside Sol's sigma window with nothing said.
    for kind in P.UNMERGED_KINDS:
        key = P.unmerged_patch_key(7, kind)
        owner = key.rsplit(".", 2)[-2].lower()
        check(f"owner of {kind} is {owner!r}", "attn" not in owner)

    print()
    if FAILURES:
        print(f"RED: {len(FAILURES)} failed -- {', '.join(FAILURES)}")
        return 1
    print("GREEN: the un-merged forward equals the merged weight, and every "
          "deliberate violation missed it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
