"""Rebalance q/k channels before INT8 attention on the blocks whose K-norm is lopsided.

## What it fixes, and where

MiniMax H3's last transformer block carries almost all of its K energy in four
channels (82, 34, 67, 19 after RoPE), because `attn.k_norm.weight` peaks there
by an order of magnitude. Both INT8 attention kernels on these graphs -- sage's
and Sol's -- quantize K per block of tokens with ONE scale across all 128
channels, so on that block four channels set the scale and the other 124 keep
about three bits. The block's attention is also the peakiest in the model
(a handful of effective keys per query on its worst heads), so a small
relative error in K becomes a large logit error and the softmax flips.
Measured: sage's INT8 error at block 49 is ~5x block 0's, and K's rounding is
almost all of it. the h3-explorations repo's `docs/attention/block49_quant_error.md` has the
anatomy; the sage fork's `CHANGELOG.md` (workload intel, "MiniMax H3, block
49") has the tables.

## What it does

For any per-channel factor s, `q . k == (q * s) . (k / s)`, so the channels
can be rebalanced before quantization at no cost to the attention math, and
Sol's routing threshold is invariant under the same rescale. H3 applies its
q/k RMSNorm weights right before RoPE, and its RoPE is split-half over
channels 0..95, rotating (i, i+48) together. A factor that is EQUAL within
each such pair commutes with the rotation, so it folds into the two norm
weights themselves: `k_norm.weight / s` and `q_norm.weight * s`. This node
adds exactly those two weight patches, per named block, through
`ModelPatcher.add_patches` -- applied when the model loads, undone when it
unloads, nothing patched in ComfyUI core, no cost at render time.

The factor comes from the checkpoint's own norm weights,
`s = |kw|^alpha / |qw|^(1-alpha)`, pair-averaged, normalized to a geometric
mean of one. That needs no capture, so it covers blocks that were never
captured, and it is neutral where the weights are flat: measured on block 0
the patched call reproduces the unpatched error to four decimals. A factor
calibrated per head removes about twice as much error at block 49 but
cannot fold into a per-channel weight; that form lives inside the sage
fork's per-thread quantizer as `qk_balance` (its v0.7.19), computed per
call, and reaches only the sage steps until Sol's quantizer gets the same.

## Off by default, and an experiment

`balance="off"` adds no patch. The other two modes are an EXPERIMENT under
`docs/attention/sol_attn.md`'s decision standard, not a shipped default: the sage-side
gain is measured on captures, the Sol-side gain is
`bench/grade_channel_balance.py`'s to establish, and nothing here is
perceptual. Which blocks are lopsided is a property of the released weights:
`bench/check_channel_balance.py` ranks all 50 from the shipped checkpoint
(measured 2026-09-14: top-4 K-norm energy share 70% at block 49, 31% at 45,
25% at 48, and 4-6% everywhere else).
"""

from __future__ import annotations

import logging
import re

import torch
from comfy_api.latest import io

ROT, HALF = 96, 48          # H3 RoPE: split-half over channels 0..95, pairs (i, i+48)
TOP_CHANNELS = 4            # the ranking statistic: energy share of the loudest four

MODES = ("off", "loud blocks (from weights)", "named blocks")


def parse_blocks(spec, count):
    """Parse "0-3,47,-1" into absolute block indices; negatives count from the end.

    `count` is the number of blocks in the model, which is what makes a
    negative index resolvable and what the result is clamped to. Ranges may be
    given either way round. Whitespace anywhere is ignored, so a spec pasted
    across lines still parses.

    Byte-identical to the definition it replaces, in the vendored Sol node and
    in `sol_attn_h3.py` -- verified by diff before the move, because a silent
    behaviour change in a parser that decides WHICH BLOCKS get different
    treatment would show up as a quality difference, not as an error.
    """
    out = set()
    for part in "".join(str(spec).split()).split(","):   # tolerate any whitespace
        if not part:
            continue
        match = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", part)
        if match is None:
            raise ValueError(f"cannot parse block spec {part!r}; "
                             "use indices and ranges like '0-3,47,-1'")
        first = int(match.group(1))
        last = first if match.group(2) is None else int(match.group(2))
        first = first if first >= 0 else count + first
        last = last if last >= 0 else count + last
        if first > last:
            first, last = last, first
        out.update(range(max(first, 0), min(last, count - 1) + 1))
    return frozenset(out)


def pair_equal_unit(s: torch.Tensor) -> torch.Tensor:
    """Make a per-channel factor RoPE-safe and scale-free.

    Geometric mean within each rotated pair so the rotation commutes with it;
    geometric mean of one over all channels so the fold changes no overall
    magnitude, only the balance between channels.
    """
    g = (s[:HALF] * s[HALF:ROT]).sqrt()
    s = torch.cat([g, g, s[ROT:]])
    return s / torch.exp(torch.log(s).mean())


def balance_factor(kw: torch.Tensor, qw: torch.Tensor, alpha: float) -> torch.Tensor:
    kw = kw.float().abs().clamp(min=1e-6)
    qw = qw.float().abs().clamp(min=1e-6)
    return pair_equal_unit(kw.pow(alpha) / qw.pow(1.0 - alpha))


def loud_share(kw: torch.Tensor) -> float:
    """Energy share of the loudest TOP_CHANNELS channels of a K-norm weight."""
    e = kw.float().pow(2)
    return (torch.topk(e, TOP_CHANNELS).values.sum() / e.sum()).item()


def weight_patches(kw: torch.Tensor, qw: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(k diff, q diff) such that k_norm.weight / s and q_norm.weight * s result.

    Returned as differences because `add_patches` composes additively with
    whatever else patches these keys, and applies them on load in fp32.
    """
    s = balance_factor(kw, qw, alpha)
    dk = (kw.float() / s - kw.float()).to(kw.dtype)
    dq = (qw.float() * s - qw.float()).to(qw.dtype)
    return dk, dq


def select_blocks(mode: str, blocks: str, share_threshold: float, k_weights: list[torch.Tensor]) -> list[int]:
    if mode == "off":
        return []
    if mode == "named blocks":
        return sorted(parse_blocks(blocks, len(k_weights)))
    return [i for i, kw in enumerate(k_weights) if loud_share(kw) >= share_threshold]


class MiniMaxH3ChannelBalance(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3ChannelBalance",
            display_name="MiniMax H3 Channel Balance",
            category="model/attention/minimax",
            is_experimental=True,
            description=(
                "Rebalances q/k channels on the transformer blocks whose K-norm "
                "weight is lopsided, so INT8 attention (sage and Sol) quantizes "
                "them with less error. Exact for the attention math; folded into "
                "the norm weights, so free at render time. Off by default; an "
                "experiment, not a default. docs/attention/sol_attn.md."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "balance", options=list(MODES), default="off",
                    tooltip=(
                        "off: no change. loud blocks (from weights): every block "
                        "whose top-4 K-norm energy share is at or above loud_share "
                        "(the shipped checkpoint gives 45, 48, 49). named blocks: "
                        "the list in blocks, dense_blocks syntax."
                    ),
                ),
                io.String.Input(
                    "blocks", default="49",
                    tooltip="Only under named blocks. e.g. '45,48,49' or '-1'; negative counts from the end.",
                ),
                io.Float.Input(
                    "alpha", default=0.5, min=0.0, max=1.0, step=0.05,
                    tooltip=(
                        "How much of the imbalance moves from K onto Q. 0.5 measured "
                        "best on block 49 (sweep in the sage fork's CHANGELOG); 1.0 "
                        "puts the whole scale on Q and measured worse everywhere."
                    ),
                ),
                io.Float.Input(
                    "loud_share", default=0.15, min=0.05, max=0.95, step=0.01,
                    tooltip=(
                        "Only under loud blocks. Threshold on the top-4 K-norm energy "
                        "share. 0.15 is reasoned from the shipped checkpoint's ranking: "
                        "loud blocks sit at 0.25-0.70, the rest at 0.04-0.06."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, balance="off", blocks="49", alpha=0.5, loud_share=0.15) -> io.NodeOutput:
        m = model.clone()
        if balance == "off":
            logging.info("[h3] channel balance: off, this node is inert")
            return io.NodeOutput(m)

        diffusion_model = model.get_model_object("diffusion_model")
        dit_blocks = getattr(diffusion_model, "blocks", None)
        if dit_blocks is None:
            raise RuntimeError(
                f"{type(diffusion_model).__name__} has no .blocks to index; "
                f"this node only patches a MiniMax H3 DiT."
            )
        k_weights = [b.attn.k_norm.weight.detach().cpu() for b in dit_blocks]
        q_weights = [b.attn.q_norm.weight.detach().cpu() for b in dit_blocks]
        wanted = select_blocks(balance, blocks, loud_share, k_weights)
        if not wanted:
            logging.info(f"[h3] channel balance: {balance!r} selected no blocks, this node is inert")
            return io.NodeOutput(m)

        patches = {}
        report = []
        for i in wanted:
            dk, dq = weight_patches(k_weights[i], q_weights[i], alpha)
            patches[f"diffusion_model.blocks.{i}.attn.k_norm.weight"] = ("diff", (dk,))
            patches[f"diffusion_model.blocks.{i}.attn.q_norm.weight"] = ("diff", (dq,))
            s = balance_factor(k_weights[i], q_weights[i], alpha)
            report.append(f"{i} (share {loud_share_of(k_weights[i]):.0%}, factor {s.min():.2f}..{s.max():.2f})")
        applied = m.add_patches(patches)
        if len(applied) != len(patches):
            missing = sorted(set(patches) - set(applied))
            raise RuntimeError(f"[h3] channel balance: {len(missing)} weight keys not in the model: {missing[:2]}...")
        logging.info(
            f"[h3] channel balance: alpha {alpha}, folded into q_norm/k_norm of "
            f"{len(wanted)} of {len(dit_blocks)} blocks: " + ", ".join(report))
        return io.NodeOutput(m)


def loud_share_of(kw: torch.Tensor) -> float:
    return loud_share(kw)
