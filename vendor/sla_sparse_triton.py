"""The sparse top-k block router and its Triton forward kernel, vendored from LightX2V.

Source: `coderef/LightX2V` at commit afcfe8f1 (2026-08-20), files
`lightx2v/common/ops/attn/utils/sla_util.py` (`compress_kernel`, `mean_pool`,
`get_block_map`) and `lightx2v/common/ops/attn/kernels/sla_kernel.py`
(`_attn_fwd`, the forward half of `_attention`). Apache-2.0, LightX2V's
LICENSE; those files are themselves LightX2V's copy of thu-ml/SLA's Triton
reference (`coderef/SLA/sparse_linear_attention/{utils,kernel}.py`), with the
linear branch removed. Copied rather than imported because the package
`__init__` pulls `loguru` and the rest of LightX2V, and because what runs here
must not move when that checkout is pulled.

## What this is, and is not

It is the attention the lightx2v Turbo-SLA LoRA was distilled under, in its
sparse-only form: mean-pool q and (k - mean k) per block, score pooled q
against pooled k, keep the top `1 - sparsity_ratio` fraction of key blocks per
query block, run flash-style attention over only those blocks. There is no
linear branch: LightX2V ships none, the paper's `proj_l` weights are in no
released H3 artifact, and the LoRA file carries none (header read
2026-08-20). Two gaps from the release's own inference path, both named in
`docs/open_experiments.md` #20: the release ran 128/64 blocks through the
SpargeAttn `sage2` operator, and this Triton path is 64/64 and bf16 -- P is
rounded to bf16 before the PV product (`p.to(v.dtype)` below), blocks are
visited in `torch.topk`'s unsorted order, so at sparsity 0 it is a flash
kernel that agrees with dense attention to bf16 accumulation, not exactly.

Forward only. The backward kernels are not copied; nothing here trains.

Changes from the source, all mechanical: the autograd `Function` is replaced
by `sparse_attn_forward`, which returns the same output; the GQA branch of
`get_block_map` is kept; nothing in the kernel body is altered.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SOURCE_COMMIT = "afcfe8f1"


@triton.jit
def compress_kernel(
    X,
    XM,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    x_offset = idx_bh * L * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    # Triton leaves masked lanes undefined when ``other`` is omitted. The
    # lanes participate in the reduction below, so zero the tail padding.
    x = tl.load(
        X + x_offset + offs_l[:, None] * D + offs_d[None, :],
        mask=offs_l[:, None] < L,
        other=0.0,
    )

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))


def mean_pool(x, BLK):
    assert x.is_contiguous()

    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    grid = (L_BLOCKS, B * H)
    compress_kernel[grid](x, x_mean, L, D, BLK)
    return x_mean


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64):
    """(sparse_map [B,H,MQ,MK] int8, lut [B,H,MQ,topk] long, topk). Layout [B,H,L,D]."""
    arg_k = k - torch.mean(k, dim=-2, keepdim=True)  # smooth-k technique in SageAttention
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)

    # GQA
    num_q_heads = q.size(1)
    num_kv_heads = k.size(1)
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0, f"Number of Q heads ({num_q_heads}) must be divisible by number of KV heads ({num_kv_heads})"
        repeat_factor = num_q_heads // num_kv_heads
        pooled_kblocks = pooled_kblocks.repeat_interleave(repeat_factor, dim=1)

    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)

    K = pooled_score.shape[-1]
    # Match the training router: short sequences still retain one key block.
    topk = max(1, min(K, int(topk_ratio * K)))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT,
    LSE,
    OS,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    qkv_offset = idx_bh * L * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    lse_offset = idx_bh * L
    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + qkv_offset + offs_n[None, :] * D + offs_d[:, None]
    V_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    OS_ptrs = OS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    LUT_ptr = LUT + lut_offset
    LSE_ptrs = LSE + lse_offset + offs_m

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L, other=0.0)
    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx)
        n_mask = offs_n < L - idx_n * BLOCK_N

        k = tl.load(
            K_ptrs + idx_n * BLOCK_N * D,
            mask=n_mask[None, :],
            other=0.0,
        )
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
        if L - idx_n * BLOCK_N < BLOCK_N:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

        v = tl.load(
            V_ptrs + idx_n * BLOCK_N * D,
            mask=n_mask[:, None],
            other=0.0,
        )
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < L)

    m_i += tl.math.log2(l_i)
    tl.store(LSE_ptrs, m_i, mask=offs_m < L)


def sparse_attn_forward(q, k, v, lut, topk, BLOCK_M=64, BLOCK_N=64, qk_scale=None):
    """Block-sparse attention over the key blocks `lut` names per query block.

    q, k, v: contiguous [B, H, L, D]; lut: contiguous [B, H, cdiv(L, BLOCK_M), topk]
    of key-block indices. Returns [B, H, L, D] in v's dtype. This is the
    forward of LightX2V's `_attention.apply` with the autograd bookkeeping
    removed.
    """
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert lut.is_contiguous()
    assert BLOCK_M in (64, 128) and BLOCK_N in (64, 128)
    B, H, L, D = q.shape
    if qk_scale is None:
        qk_scale = D ** -0.5
    M_BLOCKS = triton.cdiv(L, BLOCK_M)
    assert lut.shape == (B, H, M_BLOCKS, topk), (lut.shape, (B, H, M_BLOCKS, topk))
    o_s = torch.empty_like(v)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    grid = (M_BLOCKS, B * H)
    _attn_fwd[grid](q, k, v, qk_scale, topk, lut, lse, o_s, L, M_BLOCKS, D, BLOCK_M, BLOCK_N,
                    num_warps=4 if D == 64 else 8, num_stages=3)
    return o_s


def sla_sparse_attention(q, k, v, sparsity_ratio=0.85, BLOCK_M=64, BLOCK_N=64):
    """Router plus kernel: the one call a node or a gate makes. Layout [B, H, L, D].

    `sparsity_ratio` is the knob the release names (0.85); the router keeps
    `1 - sparsity_ratio` of key blocks, `max(1, ...)` as the training router
    did. Returned with the lut so a gate can corrupt it deliberately.
    """
    topk_ratio = 1.0 - float(sparsity_ratio)
    _, lut, topk = get_block_map(q, k, topk_ratio=topk_ratio, BLKQ=BLOCK_M, BLKK=BLOCK_N)
    lut = lut.contiguous()
    return sparse_attn_forward(q, k, v, lut, topk, BLOCK_M, BLOCK_N), lut, topk
