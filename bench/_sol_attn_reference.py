"""Sol-Attn eager reference, vendored from upstream comfy-kitchen.

Source: `comfy_kitchen/backends/eager/sol_attn.py` at `dae00a1`
(Comfy-Org/comfy-kitchen#117, the commit that merged Sol-Attn into main).
Vendored rather than imported because importing it would register a
`comfy_kitchen::sol_attn` op in a process that already has one.

DO NOT EDIT to fix a disagreement. The entire value of this file is that it
was written by the algorithm's author and not by us: if it and a kernel
disagree, that is the finding. Re-vendor from upstream instead.

Two deletions from the original, both deliberate:

  - the `from ...registry import registry` import and the
    `@torch.library.custom_op("comfy_kitchen::sol_attn")` wrapper. Importing
    those here registers an op the installed build already registers.
  - `_op_sol_attn_fake`, which exists only to serve that wrapper.

The pure function is otherwise byte-identical to upstream.

## Re-vendored 2026-08-30, from 23d1a66 to dae00a1

**The previous vendor was the pre-merge algorithm, and that is a different
claim from being out of date.** It carried `centroid_tail` and had no `tail`,
`block_len` or `coarse_gate`, so from the day the merged kernel was installed
until this re-vendor, the only controlled comparison this repo can make about
a numerical Sol knob graded the installed kernel against an algorithm that
build does not implement. It passed, because the arms exercised only the
parameters both spellings happen to share.

Free to correct, and worth knowing why: kijai's `sol_attn` branch tip and
Comfy-Org main are content-identical at this point, verified 2026-08-30 by a
whole-tree `diff -rq` between the two checkouts and by `cmp` against the
installed `site-packages`. So this file, the branch, and the running kernel
are one source.

Four upstream additions arrive with it, and the last three are one feature
rather than three knobs:

  - **`centroid_tail` is gone.** The merged kernel evaluates the pooled tail
    at the query block's centroid unconditionally, which is what `True` did,
    so the shipped configuration is unchanged and `False` is a computation
    this build cannot express.
  - **`tail`, default True.** `False` drops the pooled term entirely, leaving
    a softmax over the routed blocks only. Upstream's own tests call that
    "the SLA / VSA fine stage", which is what it is: with `topk_ratio` it is
    SLA's selection rule, and it is the shape a VSA-trained checkpoint wants.
  - **`block_len`**, an int32 count of live rows per 64-row block. It exists
    for callers that PAD -- one VSA cube per block, zero-filled to 64 -- and
    dead rows are excluded from keys, from values and from the pooled means.
    H3's packed sequence is contiguous and pads nothing, so our Sol path
    passes None and the kernel derives the ragged final block from T.
  - **`coarse_gate`**, VSA's gated coarse branch: `gate * softmax(q_mean
    k_mean^T) v_mean` added per block, where the gate is a caller-supplied
    tensor with q's exact shape. In a VSA-trained model it is a learned
    projection of the block input, so it cannot be reached from a hook that
    receives Q/K/V already built.

## Re-vendored 2026-08-22, from c04ef20 to 23d1a66

One upstream addition, and a rewrite underneath it that this file cannot see:

  - **`topk_ratio`, default 0.0.** Above zero it replaces the tau threshold
    with SLA-style per-query-block top-k. It changes SELECTION ONLY: the
    pooled tail still runs for every unselected block, which is what
    distinguishes it from `MiniMaxH3SLARouter`, where unselected blocks are
    dropped outright. At the shipped default of 0.0 this path is not taken,
    so it does not change what the oracle says about any render here.
    **Since the merge that is no longer the only difference** -- `tail=False`
    now expresses "drop them outright" directly.
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
    the node exposed it defaulting True, so between ad9a4a8 and now the
    vendored oracle did not contain the path a real render would take. Any
    correctness verdict from the old file describes `centroid_tail=False`
    only. **Superseded by the 2026-08-30 re-vendor**, which removed the
    argument along with the choice.
  - **`key_bias`**, a per-key logit bias for mask-like inputs. Unused by our
    Sol path -- the node declines masked attention outright (`_ineligible`
    returns "masked attention") -- and legal only where the biased blocks are
    sink-covered, since the pooled tail cannot see a per-token bias.

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


def _block_lengths(t: int, n: int, device, block_len=None) -> torch.Tensor:
    """Live tokens per block: the caller's table (zero-padded tiles) or 64
    everywhere except a ragged last block."""
    tail = t - (n - 1) * BLOCK
    if block_len is not None:   # clamped like the kernels (block_len_of)
        lengths = block_len.to(device=device, dtype=torch.float32).clamp(1, BLOCK)
        if tail < BLOCK:
            lengths = lengths.clone()
            lengths[-1] = lengths[-1].clamp(max=tail)
        return lengths
    lengths = torch.full((n,), float(BLOCK), device=device)
    if tail < BLOCK:
        lengths[-1] = float(tail)
    return lengths


def _sink_count(n, s0, s1):
    """Sink blocks inside [0, n): they are always exact, so the top-k budget
    counts the others only."""
    return max(0, min(s1, n) - min(s0, n))


def _valid_rows(t: int, lengths: torch.Tensor) -> torch.Tensor:
    """(T,) bool: token t is live iff it sits in the first len rows of its block."""
    pos = torch.arange(t, device=lengths.device)
    return (pos % BLOCK) < lengths[pos // BLOCK]


def coarse_output(qm, km, vm, scale):
    """VSA coarse branch: dense attention over the block means. qm/km/vm are
    ``[BH, N, D]`` fp32; returns ``[BH, N, D]`` fp32."""
    s = torch.bmm(qm, km.transpose(1, 2)) * scale
    return torch.bmm(torch.softmax(s, dim=-1), vm)


def add_coarse_(out, oc, gate):
    """out += gate * oc[block(t)], in place (one rounding: addcmul_ promotes
    internally). out/gate ``(B, T, H, D)``; oc ``[BH, N, D]`` fp32."""
    b, t, h, d = out.shape
    oc = oc.view(b, h, -1, d).permute(0, 2, 1, 3)   # (B, N, H, D)
    gate = gate.contiguous()
    nfull = t // BLOCK
    if nfull:
        out[:, :nfull * BLOCK].view(b, nfull, BLOCK, h, d).addcmul_(
            gate[:, :nfull * BLOCK].view(b, nfull, BLOCK, h, d), oc[:, :nfull, None])
    if t % BLOCK:
        out[:, nfull * BLOCK:].addcmul_(gate[:, nfull * BLOCK:], oc[:, nfull:nfull + 1])
    return out


def _topk_count(n: int, ratio: float) -> int:
    """Key blocks a query block keeps under top-k: ratio * n, clamped to
    [1, n-1]; 0 when n <= 1 (nothing to select beyond the forced blocks)."""
    return max(0, min(n - 1, max(1, round(ratio * n))))


def _pool(x: torch.Tensor, n_blocks: int, reduce: str, lengths=None) -> torch.Tensor:
    """(B, T, H, D) -> (B, N, H, D), block mean or sum over the live rows."""
    b, t, h, d = x.shape
    if lengths is None:
        lengths = _block_lengths(t, n_blocks, x.device)
    x = x * _valid_rows(t, lengths).view(1, -1, 1, 1)
    pad = n_blocks * BLOCK - t
    if pad:
        x = torch.cat([x, x.new_zeros(b, pad, h, d)], dim=1)
    blocks = x.reshape(b, n_blocks, BLOCK, h, d)
    if reduce == "sum":
        return blocks.sum(dim=2)
    return blocks.sum(dim=2) / lengths.view(1, -1, 1, 1)


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float = 1.0,
    scale: float | None = None,
    sink_blocks: list[int] | None = None,
    sink_q: list[int] | None = None,
    key_bias: torch.Tensor | None = None,
    topk_ratio: float = 0.0,
    tail: bool = True,
    block_len: torch.Tensor | None = None,
    coarse_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sol-Attn over ``(B, T, H, D)`` tensors. See the module docstring.

    ``topk_ratio`` > 0 selects SLA-style per-query-block top-k instead of the
    tau threshold (sinks and diagonal still forced exact); tau is ignored.
    ``tail=False`` drops the pooled term (softmax over routed blocks only),
    ``block_len`` marks the live rows at the front of each 64-block (values
    clamped to [1, rows in the block]; dead rows are never keys and their
    output rows are unspecified), and ``coarse_gate`` adds VSA's gated coarse
    branch: ``gate * softmax(q_mean k_mean^T * scale) v_mean`` per block.
    """
    b, t, h, d = q.shape
    n = (t + BLOCK - 1) // BLOCK
    if scale is None:
        scale = d ** -0.5
    log2s = scale * _LOG2E
    sink_kv0, sink_kv1 = (sink_blocks or [0, 0])[:2]
    sink_q0, sink_q1 = (sink_q or [0, 0])[:2]

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
    lengths = _block_lengths(t, n, q.device, block_len)
    kc = _pool(fk, n, "mean", lengths)              # (B, N, H, D) summary keys
    vc = _pool(fv, n, "sum", lengths)               # (B, N, H, D) summed values

    # centring K shifts every score in a row by a constant: softmax-invariant
    k_mean = kc.mean(dim=1, keepdim=True)           # (B, 1, H, D)
    kcc = kc - k_mean
    kc_var = kcc.pow(2).mean(dim=1)                 # (B, H, D)

    centroid = _pool(fq, n, "mean", lengths)                        # (B, N, H, D)

    qh = fq.permute(0, 2, 1, 3)                                     # (B, H, T, D)
    kh = (fk - k_mean).permute(0, 2, 1, 3)
    vh = fv.permute(0, 2, 1, 3)
    kch = kcc.permute(0, 2, 1, 3)                                   # (B, H, N, D)
    vch = vc.permute(0, 2, 1, 3)

    s_tok = (qh @ kh.transpose(-1, -2)) * log2s                     # (B, H, T, T)
    if block_len is not None:
        s_tok = s_tok.masked_fill(~_valid_rows(t, lengths).view(1, 1, 1, t),
                                  torch.finfo(s_tok.dtype).min)
    if key_bias is not None:
        # Per-key logit bias (natural log). Exact branch only: biased blocks
        # must be sink-covered, the pooled tail cannot see per-token bias.
        kb = _normalize_key_bias(key_bias, b, t, q.device)
        s_tok = s_tok + (kb * _LOG2E).reshape(-1, 1, 1, t)
    s_blk = (qh @ kch.transpose(-1, -2)) * log2s                    # (B, H, T, N)

    # routed = column mean over the query block clears the threshold; the
    # diagonal +-1, sink blocks and sink_q rows are always exact
    qblk = torch.arange(t, device=q.device) // BLOCK
    if block_len is not None:   # dead query rows must not enter the block's mean
        s_blk = s_blk * _valid_rows(t, lengths).view(1, 1, t, 1)
    colmean = torch.zeros(b, h, n, n, device=q.device, dtype=s_blk.dtype)
    colmean.scatter_add_(2, qblk.view(1, 1, t, 1).expand(b, h, t, n), s_blk)
    colmean = colmean / lengths.view(1, 1, n, 1)
    idx = torch.arange(n, device=q.device)
    if topk_ratio:
        # sink blocks are always exact, so they neither count toward nor consume the budget
        ranked = colmean.clone()
        ranked[..., sink_kv0:sink_kv1] = float("-inf")
        kk = _topk_count(n - _sink_count(n, sink_kv0, sink_kv1), topk_ratio)
        # >= the k-th score: a tied group at the boundary is kept, not dropped
        if kk:
            row_thr = ranked.topk(kk, dim=-1).values[..., -1:]
            exact = ranked >= row_thr
        else:
            exact = torch.zeros_like(ranked, dtype=torch.bool)
    else:
        # tau sigma of the proxy row, from the query-block centroid
        var = (centroid.pow(2) * kc_var.unsqueeze(1)).sum(-1)      # (B, N, H)
        thr = tau * torch.sqrt(var * log2s * log2s + 1e-6)
        exact = colmean > thr.permute(0, 2, 1).unsqueeze(-1)            # (B, H, NQ, N)
    exact |= ((idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1).view(1, 1, n, n)
    exact |= ((idx >= sink_kv0) & (idx < sink_kv1)).view(1, 1, 1, n)
    exact |= ((idx >= sink_q0) & (idx < sink_q1)).view(1, 1, n, 1)

    ex_tok = exact.gather(2, qblk.view(1, 1, t, 1).expand(b, h, t, n))   # (B,H,T,N)
    keep_tok = ex_tok.repeat_interleave(BLOCK, dim=-1)[..., :t]
    neg = torch.finfo(s_tok.dtype).min
    s_tok = s_tok.masked_fill(~keep_tok, neg)
    # every row shares its query block's tail (colmean IS the centroid score)
    s_blk = colmean.gather(2, qblk.view(1, 1, t, 1).expand(b, h, t, n))
    s_blk = s_blk.masked_fill(ex_tok, neg)
    if not tail:
        s_blk = torch.full_like(s_blk, neg)

    # one softmax over both branches; a pooled term weighs its block length (vc is a sum)
    logits = torch.cat([s_tok, s_blk], dim=-1)
    p = torch.exp2(logits - logits.amax(dim=-1, keepdim=True))
    p = p.masked_fill(logits <= neg, 0.0)
    num = p[..., :t] @ vh + p[..., t:] @ vch
    den = p[..., :t].sum(-1) + (p[..., t:] * lengths.view(1, 1, 1, n)).sum(-1)
    out = (num / den.clamp_min(1e-30).unsqueeze(-1)).permute(0, 2, 1, 3).to(v.dtype)
    # contiguous, matching the CUDA backend and register_fake
    out = out.contiguous()
    if coarse_gate is not None:
        flat = lambda p: p.permute(0, 2, 1, 3).reshape(b * h, n, d)  # noqa: E731
        oc = coarse_output(flat(centroid), flat(kc), flat(vc / lengths.view(1, -1, 1, 1)), scale)
        add_coarse_(out, oc, coarse_gate)
    return out
