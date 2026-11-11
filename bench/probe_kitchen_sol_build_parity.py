#!/usr/bin/env python3
"""Hash `sol_attn` and `sol_attn_chunked` outputs over a grid of options, so two comfy-kitchen builds can be diffed line by line.

One line per case on stdout: the options, then a hash of the output's bytes.
Run it once per build and `diff` the two files; a line that differs is a case
the two builds compute differently, and no differing line means bit-identical
output on every case in the grid. The build's version goes to stderr, so the
two files diff clean when the builds agree.

The grid: bfloat16 and float16; several sequence lengths, one of them ragged;
random inputs and inputs with two loud K channels; `rotate` off and on; the
tau threshold, every block routed, top-k, no tail, sinks, and token routing at
two budgets; then the chunked producer at two shapes, bootstrap and with the
previous step's statistics. It passes `rotate` only when it is on and no other
fork option, so it runs on any build that has `rotate`.

What it cannot see: anything off the grid, in particular `qk_balance`,
`tau_map` and `blk_cnt`, and inputs where a token list overflows (random
inputs do not). It says two builds agree; it does not say either is right.

Needs the GPU and no server. To run a build that is not the installed one,
unpack its wheel somewhere and put that folder first on `PYTHONPATH`.

    <comfy-venv-python> bench/probe_kitchen_sol_build_parity.py > a.txt
    PYTHONPATH=<unpacked wheel> <comfy-venv-python> bench/probe_kitchen_sol_build_parity.py > b.txt
    diff a.txt b.txt
"""
import hashlib, importlib.metadata, sys
import torch
import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as cb

def sha(x):
    return hashlib.sha256(x.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]

def qkv(t, h, seed, dtype, loud):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = (torch.randn(1, t, h, 128, device="cuda", generator=g) for _ in range(3))
    if loud:
        k[..., 82] *= 12.0; k[..., 19] *= 12.0
    return tuple(x.to(dtype).contiguous() for x in (q, k, v))

print("# build", importlib.metadata.version("comfy_kitchen"), file=sys.stderr)
for dtype in (torch.bfloat16, torch.float16):
    for t, h in ((1024, 4), (3137, 4), (8192, 8), (20000, 8)):
        for loud in (False, True):
            q, k, v = qkv(t, h, t + h, dtype, loud)
            for rotate in (False, True):
                for name, kw in (("tau1", dict(tau=1.0)), ("all", dict(tau=-1e9)), ("topk", dict(topk_ratio=0.2)),
                                 ("notail", dict(tau=1.0, tail=False)),
                                 ("sinks", dict(tau=1.0, sink_blocks=[0, 2], sink_q=[1, 3])),
                                 ("tok64", dict(tau=1.0, token_aug=64)), ("tok256sink", dict(tau=2.0, token_aug=256, sink_blocks=[0, 1]))):
                    extra = dict(rotate=True) if rotate else {}
                    out = ck.sol_attn(q, k, v, **kw, **extra)
                    print(f"direct {str(dtype)[6:]:8s} T={t:5d} H={h} loud={int(loud)} rotate={int(rotate)} {name:10s} {sha(out)}")
# chunked producer (bf16, B=1)
for seed, t, h, rot in ((23, 4096, 4, 96), (5, 3137, 2, 64)):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qkvp = torch.randn(t, 3 * h * 128, device="cuda", dtype=torch.bfloat16, generator=g)
    freqs = torch.randn(1, t, 1, rot // 2, 2, 2, device="cuda", generator=g)
    qw = torch.randn(128, device="cuda", dtype=torch.bfloat16, generator=g)
    kw_ = torch.randn(128, device="cuda", dtype=torch.bfloat16, generator=g)
    chunks = list(qkvp.split(1024))
    for rotate in (False, True):
        for name, kw in (("tau1", dict(tau=1.0)), ("all", dict(tau=-1e9)), ("tok64", dict(tau=1.0, token_aug=64))):
            extra = dict(rotate=True) if rotate else {}
            out, km, vs = cb.sol_attn_chunked(chunks, t, h, freqs, (qw, kw_), **kw, **extra)
            out2, _, _ = cb.sol_attn_chunked(chunks, t, h, freqs, (qw, kw_), kmean=km, vscale=vs, **kw, **extra)
            print(f"chunked T={t:5d} H={h} rot={rot} rotate={int(rotate)} {name:6s} {sha(out)} {sha(out2)}")
