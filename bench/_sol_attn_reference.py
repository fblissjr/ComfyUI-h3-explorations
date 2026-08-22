"""Sol-Attn eager reference, vendored from kijai/comfy-kitchen.

Source: branch `sol_attn`, commit 23d1a66, file
`comfy_kitchen/backends/eager/sol_attn.py`. Vendored rather than imported
because that branch is unmerged and ships no wheel on PyPI -- the stock
comfy-kitchen (0.2.31) has no `sol_attn` at all. A local build of the branch
does, and `bench/check_sol_kernel.py` is what tells the two apart.

DO NOT EDIT to fix a disagreement. The entire value of this file is that it
was written by the algorithm's author and not by us: if it and a kernel
disagree, that is the finding. Re-vendor from upstream instead.

Two deletions from the original, both deliberate:

  - the `from ...registry import registry` import and the
    `@torch.library.custom_op("comfy_kitchen::sol_attn")` wrapper. Importing
    those here would register a `comfy_kitchen::sol_attn` op in the running
    process, which now really does collide: a local build of the branch is
    installed and registers that op itself.
  - `_accepts_max_blocks`, which only existed to feed that wrapper. Its
    `functools` and `inspect` imports go with it.

The pure function is otherwise byte-identical to upstream.

## Re-vendored 2026-08-22, from c04ef20 to 23d1a66

One upstream addition, and a rewrite underneath it that this file cannot see:

  - **`topk_ratio`, default 0.0.** Above zero it replaces the tau threshold
    with SLA-style per-query-block top-k. It changes SELECTION ONLY: the
    pooled tail still runs for every unselected block, which is what
    distinguishes it from `MiniMaxH3SLARouter`, where unselected blocks are
    dropped outright. At the shipped default of 0.0 this path is not taken,
    so it does not change what the oracle says about any render here.
  - Upstream also rewrote the CUDA routing kernel in this range
    (`sage_attention/sol_attn_route.cu`, "Optimize routing"). Nothing of that
    is visible in this file -- which is the point of grading the kernel
    against it rather than against itself.

## Re-vendored 2026-08-14, from ad9a4a8 to c04ef20

The previous vendor predated two upstream additions, and the first one is not
cosmetic:

  - **`centroid_tail`, default True.** Every row shares its query block's
    centroid tail rather than computing its own. Upstream measures ~5e-4
    cosine against per-row for 64x less routing work. It is the DEFAULT, and
    `internal/refs/sol_attn_minimax.py` exposes it defaulting True, so between
    ad9a4a8 and now the vendored oracle did not contain the path a real render
    would take. Any correctness verdict from the old file describes
    `centroid_tail=False` only.
  - **`key_bias`**, a per-key logit bias for mask-like inputs. Unused by us --
    the H3 node declines masked attention outright (`_ineligible` returns
    "masked attention") -- so it is dead weight here, kept only because
    editing the vendored function is what this file forbids.

Upstream also reports correctness fixes on the CUDA side in this range
(reported by the author, not verified here). That is a further reason the old
vendor could not adjudicate: it was the pre-fix algorithm.

It is O(T^2) and materialises the full score tensor, so it refuses past 4 GiB
-- it cannot run at H3's real sequence length. Small shapes only.
"""

import torch


BLOCK = 64
_LOG2E = 1.4426950408889634
# Past this the caller almost certainly wanted a fused backend and got
# silently downgraded; a clear error beats an allocator failure.
_MAX_SCORE_BYTES = 4 * 2**30


def _normalize_key_bias(key_bias, batch, t, device):
    """Reduce SDPA-mask-like forms -- (T,), (B, T), (B|1, 1, 1, T), bool or
    float -- to (B, T) float log-bias. Rejects head/query-varying masks and
    wrong-device tensors (a host pointer would poison the CUDA context).
    """
    if key_bias.device != device:
        raise ValueError(
            f"sol_attn: key_bias must be on {device}, got {key_bias.device}")
    if key_bias.dim() == 4:
        if key_bias.shape[1] != 1 or key_bias.shape[2] != 1:
            raise ValueError(
                "sol_attn: key_bias must be key-only; a mask varying over "
                f"heads or queries ({tuple(key_bias.shape)}) cannot be "
                "expressed by this op -- use a dense attention for those calls")
        key_bias = key_bias[:, 0, 0, :]
    if key_bias.dim() == 1:
        key_bias = key_bias.unsqueeze(0)
    if key_bias.dim() != 2 or key_bias.shape[-1] != t or key_bias.shape[0] not in (1, batch):
        raise ValueError(
            f"sol_attn: key_bias must be (T,), (B, T) or (B, 1, 1, T), got "
            f"{tuple(key_bias.shape)} for T={t}, B={batch}")
    if key_bias.dtype == torch.bool:
        key_bias = torch.where(key_bias, 0.0, float("-inf"))
    return key_bias.float()


def _pool(x: torch.Tensor, n_blocks: int, reduce: str) -> torch.Tensor:
    """(B, T, H, D) -> (B, N, H, D), block mean or sum, ragged tail handled."""
    b, t, h, d = x.shape
    pad = n_blocks * BLOCK - t
    if pad:
        x = torch.cat([x, x.new_zeros(b, pad, h, d)], dim=1)
    blocks = x.reshape(b, n_blocks, BLOCK, h, d)
    if reduce == "sum":
        return blocks.sum(dim=2)
    lengths = x.new_full((n_blocks,), float(BLOCK))
    if pad:
        lengths[-1] = float(BLOCK - pad)
    return blocks.sum(dim=2) / lengths.view(1, -1, 1, 1)


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float = 1.0,
    scale: float | None = None,
    sink_blocks: list[int] | None = None,
    sink_q: list[int] | None = None,
    centroid_tail: bool = True,
    key_bias: torch.Tensor | None = None,
    topk_ratio: float = 0.0,
) -> torch.Tensor:
    """Sol-Attn over ``(B, T, H, D)`` tensors. See the module docstring.

    ``topk_ratio`` > 0 selects SLA-style per-query-block top-k instead of the
    tau threshold (sinks and diagonal still forced exact); tau is ignored.
    """
    b, t, h, d = q.shape
    n = (t + BLOCK - 1) // BLOCK
    if scale is None:
        scale = d ** -0.5
    log2s = scale * _LOG2E
    sink_kv0, sink_kv1 = (sink_blocks or [0, 0])[:2]
    sink_q0, sink_q1 = (sink_q or [0, 0])[:2]

    # Anything routed here by accident (fp16/fp32 input at video length) would
    # otherwise die in the allocator with nothing pointing at the cause.
    score_bytes = b * h * t * t * 4
    if score_bytes > _MAX_SCORE_BYTES:
        raise RuntimeError(
            f"sol_attn: the eager reference is O(T^2) and would need "
            f"{score_bytes / 2**30:.1f} GiB for the score tensor at "
            f"(B={b}, H={h}, T={t}). It was selected because no fused backend "
            f"accepted these inputs -- the CUDA backend requires bfloat16 on CUDA "
            f"with head_dim 128, and got {q.dtype} on {q.device.type}."
        )

    fq, fk, fv = q.float(), k.float(), v.float()
    kc = _pool(fk, n, "mean")                       # (B, N, H, D) summary keys
    vc = _pool(fv, n, "sum")                        # (B, N, H, D) summed values

    # Centring K by the pooled mean shifts every score in a row by a constant,
    # leaving the softmax unchanged; it only shrinks the INT8 dynamic range.
    k_mean = kc.mean(dim=1, keepdim=True)           # (B, 1, H, D)
    kcc = kc - k_mean
    kc_var = kcc.pow(2).mean(dim=1)                 # (B, H, D)

    # Routing threshold: tau sigma of the proxy row, from the query-block centroid.
    lengths = fq.new_full((n,), float(BLOCK))
    if n * BLOCK - t:
        lengths[-1] = float(t - (n - 1) * BLOCK)
    centroid = _pool(fq, n, "mean")                                 # (B, N, H, D)
    var = (centroid.pow(2) * kc_var.unsqueeze(1)).sum(-1)          # (B, N, H)
    thr = tau * torch.sqrt(var * log2s * log2s + 1e-6)

    qh = fq.permute(0, 2, 1, 3)                                     # (B, H, T, D)
    kh = (fk - k_mean.squeeze(1).unsqueeze(1)).permute(0, 2, 1, 3)
    vh = fv.permute(0, 2, 1, 3)
    kch = kcc.permute(0, 2, 1, 3)                                   # (B, H, N, D)
    vch = vc.permute(0, 2, 1, 3)

    s_tok = (qh @ kh.transpose(-1, -2)) * log2s                     # (B, H, T, T)
    if key_bias is not None:
        # Per-key logit bias (natural log). Exact branch only: biased blocks
        # must be sink-covered, the pooled tail cannot see per-token bias.
        kb = _normalize_key_bias(key_bias, b, t, q.device)
        s_tok = s_tok + (kb * _LOG2E).reshape(-1, 1, 1, t)
    s_blk = (qh @ kch.transpose(-1, -2)) * log2s                    # (B, H, T, N)

    # A block is routed if its mean score over the query block clears the
    # threshold; the diagonal and its neighbours, and any sink block, are always
    # exact, and every block is exact for a query inside the sink range.
    qblk = torch.arange(t, device=q.device) // BLOCK
    colmean = torch.zeros(b, h, n, n, device=q.device, dtype=s_blk.dtype)
    colmean.scatter_add_(2, qblk.view(1, 1, t, 1).expand(b, h, t, n), s_blk)
    colmean = colmean / lengths.view(1, 1, n, 1)
    idx = torch.arange(n, device=q.device)
    if topk_ratio:
        kk = max(1, min(n - 1, round(topk_ratio * n)))
        row_thr = colmean.topk(kk + 1, dim=-1).values[..., -1:]
        exact = colmean > row_thr
    else:
        exact = colmean > thr.permute(0, 2, 1).unsqueeze(-1)            # (B, H, NQ, N)
    exact |= ((idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1).view(1, 1, n, n)
    exact |= ((idx >= sink_kv0) & (idx < sink_kv1)).view(1, 1, 1, n)
    exact |= ((idx >= sink_q0) & (idx < sink_q1)).view(1, 1, n, 1)
    valid_blk = (idx * BLOCK < t).view(1, 1, 1, n)
    exact &= valid_blk

    ex_tok = exact.gather(2, qblk.view(1, 1, t, 1).expand(b, h, t, n))   # (B,H,T,N)
    keep_tok = ex_tok.repeat_interleave(BLOCK, dim=-1)[..., :t]
    neg = torch.finfo(s_tok.dtype).min
    s_tok = s_tok.masked_fill(~keep_tok, neg)
    if centroid_tail:
        # Every row shares its query-block centroid's tail (colmean IS the
        # centroid score); ~5e-4 cosine vs per-row, 64x less routing work.
        s_blk = colmean.gather(2, qblk.view(1, 1, t, 1).expand(b, h, t, n))
    s_blk = s_blk.masked_fill(ex_tok | ~valid_blk, neg)

    # One softmax over both branches. A pooled term carries its block's length in
    # the denominator because vc is a sum, not a mean.
    logits = torch.cat([s_tok, s_blk], dim=-1)
    p = torch.exp2(logits - logits.amax(dim=-1, keepdim=True))
    p = p.masked_fill(logits <= neg, 0.0)
    num = p[..., :t] @ vh + p[..., t:] @ vch
    den = p[..., :t].sum(-1) + (p[..., t:] * lengths.view(1, 1, 1, n)).sum(-1)
    out = (num / den.clamp_min(1e-30).unsqueeze(-1)).permute(0, 2, 1, 3).to(v.dtype)
    # Contiguous to match the CUDA backend and register_fake (torch.compile
    # plans downstream ops against the fake's strides).
    return out.contiguous()
