# Cheap scorers that choose what attention computes exactly, finer than a 64-token block (2026 sweep)

> **Provenance, 2026-09-19.** Written by a research subagent of the `libguy`
> session for the lead's wider sweep, read-only, no GPU. The commissioning
> session re-checked its load-bearing pointers and found them correct: the
> `token_aug` passes in
> `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu:17-29`
> (two QK passes over unrouted keys, the second also folding the tail),
> the per-warp query rows and shared key tile in
> `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_exact.cu:40,86,154`,
> the per-warp tile skip at `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu:277-281`,
> the node defaults at `sol_attn_h3.py:1477,1530,1617-1619`, sglang's
> SubBlock table at
> `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:27-33`,
> and MiniMax-M3's max/LSE block score at
> `coderef/sglang/python/sglang/kernels/ops/attention/minimax_sparse/prefill/flash_with_topk_idx.py:231-239`.
> Paper figures came through a summarising fetch tool, as the agent says
> below; recheck against the PDF before reusing one. `bench/analyze_sol_block_grouping.py`
> was a peer session's uncommitted work when this was written. Its section 5
> cost for `token_aug` supersedes the single-pass figure in
> [`2026-09-19_dsv41_hierarchical_indexer.md`](2026-09-19_dsv41_hierarchical_indexer.md)
> section 3.1.

**Provenance.** Written by a research subagent on 2026-09-19. Read-only: no git, no GPU, nothing edited in the repo or in `coderef/`. **Local code read in full or at the cited lines:**
- the sglang SubBlock router (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py`);
- sglang's QSA package, MiniMax-M3 sparse backend, MiniCPM sparse utilities and VMoBA;
- flashinfer's `prims_ts` README and `msa_ops`;
- comfy-kitchen's Sol route, token and exact kernels (header comments, constants and the lines cited);
- this repo's `workflows/h3_config.py` Sol defaults and the header and verdict text of `bench/analyze_sol_block_grouping.py`. That tool is uncommitted peer work in the shared tree as of this draft.

**Read through a fetch tool that summarises the HTML** (so the section and table pointers are as the tool reported them, and any figure should be rechecked against the PDF before it is reused): HISA, PIVOT, Self-Indexing Attention, RaBitQCache, MiniMax Sparse Attention, DFSAttn, FG-Attn, XAttention, CompactAttention, Token Sparse Attention, SpotAttention, Stem, HAWK and Trend-aware Pruning.

**One external source read as code:** DFSAttn's `dfsattn/attention_wan.py`, downloaded from GitHub to the scratchpad.

**Read as abstracts only:** ParisKV, IndexCache, MISA, AsyncTLS, ProxyAttn, MOD-DiT, SinkPruner, TOPS, MaMe/MaRe, LTBM, Stride-k, TileMaxSim, Multipole Attention and every 2024–2025 base paper. I verified every arXiv id cited below against arxiv.org (abstract page or the arXiv API) on 2026-09-19.

**Scope.** Out of scope, and covered by other agents: DeepSeek's own indexer (V3.2/V4.x, CSA2), and LoSA, "Attention Sparsity is Input-Stable" (SVOO), HASTE and cumulative-energy filtering.

---

## The answer, first screen

- **[V] The token stage the team is asking for already exists in the kernel on this box, and it is off.** It is kitchen's `token_aug`. It runs one centroid per two query blocks (128 query rows, except a ragged final group) and scores every individual key in the blocks none of them routed, on INT8 MMA. It admits whole histogram bins, gathers the listed keys into tiles for the exact kernel, and makes the rest of the tail exact for the centroid. Sources:
  - `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu:17-29`
  - `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh:44-45` (`TOK_GROUP = 2`)
  - `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_exact.cu:178`

  It is off both in the shipped graphs (`workflows/h3_config.py`, the `token_aug_blocks=""` block and its comment) and in the node's own schema defaults (`sol_attn_h3.py:1477`, `token_aug_blocks` default empty; `sol_attn_h3.py:1617-1619`, `token_routing` default "text field"). So the design question is not whether to add a second stage. It is which side of the 64×64 decision to refine, and with what summary.
- **[V/I] Its geometry is the least favoured one.** `token_aug` is coarse on the query side (128 rows) and exact on the key side (1 token). The two pieces of local evidence both point at the query side and at a max-like key summary:
  - The peer tool's verdict (`bench/analyze_sol_block_grouping.py`, the `VERDICT` text) finds the headroom on the QUERY side: one decision per 64 queries is not right for all of them. It also finds that ranking key blocks by their true maximum beats the centroid rule at the same block count.
  - sglang measured on H3 that splitting the query side alone, when the pieces are recombined into ONE decision, is worse than not splitting, and that splitting both sides is best (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:27-44`).
- **[I, arithmetic from shapes] A symmetric 16×16 sub-block scorer costs about a fifth of what `token_aug` costs, or about a third with the pooled tail off.** At 1,600 key blocks per head and head_dim 128, a 16×16 pooled-cell router is 1/512 of dense attention. `token_aug` makes three d-wide GEMM passes over every unrouted token at one centroid per 128 queries: INT8 QK for the histogram, INT8 QK again to list tokens, then softmax·PV for the tail (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu:283-288,338-343`). The `pooled_tail` default is on (`sol_attn_h3.py:1530`). That makes it 3u/256 of dense, where u is the unrouted share, or about 1/107 at u = 0.8. With the tail off it is 2u/256, about 1/160. So the finer router is the cheaper stage, and neither needs reuse across steps to be affordable. The full table is in §5.
- **[V] The 2026 LLM work that transfers is the search path and the kernel contract, not the indexers.** Every production LLM indexer surveyed is either learned (QSA, MiniMax MSA, SpotAttention) or trained into its model (NSA, InfLLM-v2, MoBA). The training-free 2026 results rewrite how a learned indexer's candidates are searched:
  - hierarchical block-then-token (HISA, 2603.28458);
  - one proxy query per query group, then re-scoring per query (PIVOT, 2607.24593);
  - routing to a few index heads (MISA, 2605.07363);
  - cross-layer index reuse (IndexCache, 2603.12201).

  On the execution side, flashinfer and MiniMax both run per-query-token block lists as one online softmax per query tile over the UNION of the tile's blocks. That is the contract a query-split Sol would need. It is Blackwell-only in both libraries.
- **[V] Retrieval-style scoring (PQ, LSH, clustering) pays for its index over many decode steps. H3 has no decode loop and rebuilds K every block and step** (`docs/research/technique_transfer.md`, facts 1 and 2). Only one-pass key sketches survive that: sign bits after a Hadamard (Self-Indexing Attention, 2609.13205), RaBitQ-style 1-bit keys (2606.31519) and per-block summaries. The Quest bound, the best-known per-block bound, was already tested on H3 captures by the peer tool (arm D), and it ranks worse than the centroid rule: too loose at head_dim 128 over 64 keys.
- **[V] The one sub-block video scorer with public code, DFSAttn (2605.23445), uses the estimator sglang's H3 notes rejected in pixels.** It takes a softmax over key tiles per query tile, summed per block pair. That is "normalise each query sub-block into a distribution", which sglang measured better on two offline proxies and worse on 0 of 15 prompts end to end (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:48-61`). Treat DFSAttn as a warning about offline proxies, not as a recipe.

## What this means for H3, ranked

Ranked by how directly each item could change a Sol default or a kitchen kernel within a week. Every "measure first" is offline on the existing captures, run as new arms beside the peer tool's B/C/D/E rows. The peer owns the tool, so coordinate before editing it. Compare at **equal executed cost**, not equal tau: the tool's own header warns that equal-tau cells route at different densities.

1. **Split the query side into four warp-sized sub-blocks that route separately against the same 64-key blocks.** [I, design; V, evidence]
   - **Why it ranks first.** It is the one change with local evidence (the peer tool's C rows and per-query oracle) and a production kernel contract to copy (flashinfer QToken-KvBlock and MSA, §4.1).
   - **Why it maps onto the kernel.** [V] Kitchen's exact kernel splits the QUERY axis across its four warps. Each warp owns 16 query rows (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_exact.cu:40,86`: `NWARP = BQ / 16`, `q_row0 = q_block * BQ + warp * 16 + g`). Every warp runs its own Q fragment against the whole shared key tile (`:154`, `tile_qk(SK(cur), qa, ...)`), and the warps meet at `__syncthreads()` around each shared stage (`:171-174`). The token kernel already skips a tile per warp on a warp-uniform `__all_sync` test while still joining the barrier (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu:277-281`). [I] So a per-16-row decision is one warp's rows. The CTA would walk the union of its four warps' lists, and each warp would skip the tiles outside its own set the same way. That saves MMA work, not tile loads.
   - **Cost.** 4× the route's QK, plus 4× the pooled-tail PV if each warp keeps its own tail: 1/1024 of dense (§5).
   - **Not refuted by sglang's negative.** sglang's "split Q alone is worse" row recombined the sub-blocks into ONE decision per 64 rows by log-sum-exp (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:31,37-41`). Separate decisions are a different estimator, and nothing local has measured them.
   - **Measure first.** A `split4` arm: each 16-row sub-block routes with the same tau-sigma rule from its own 16-row centroid. Report B(i), the missed mass, per sub-block. Also report the **union inflation**: the union's tile count per 64-block divided by the mean per-warp count, because the union is what the CTA loads. CompactAttention measured this inflation on an LLM (§4.1). PIVOT measured a group-of-4 union of 1.3–1.5× k on DeepSeek-V3.2 (its Figure 2, as the fetch tool reported it).

2. **Replace the key block's mean with a max-like summary built from four 16-key sub-means: a log-sum-exp (SubBlock's form) or a plain max.** [I, design; V, evidence]
   - **Route-only change.** It is confined to the route kernel. The exact kernel and the tile layout do not change.
   - **Evidence.** The peer tool's `blockmax` arm says the max objective beats the mean (`VERDICT`: "the objective is fine and the relaxation is not"). SubBlock's H3 table shows n_k = 4 moving retained-mass recall part of the way from plain pooling toward its oracle at fixed 0.9 sparsity (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:27-33`; sglang's measurement on their hardware). MiniMax-M3's indexer scores a block by the MAX of its exact per-token logits, with LSE as a config option (`coderef/sglang/python/sglang/kernels/ops/attention/minimax_sparse/prefill/flash_with_topk_idx.py:231-239`, `coderef/sglang/python/sglang/srt/configs/model_config.py:340-346`). MiniCPM's InfLLM-v2 max-pools compressed-key scores into blocks (`coderef/sglang/python/sglang/srt/layers/attention/minicpm/sparse_utils.py:216-296`). Both of those are trained models.
   - **Cost.** 4 G²d: 1/2048 of dense.
   - **Measure first.** A `lse4` arm and a `max4` arm. Rank key blocks by the LSE (or max) over the four sub-means against the query-block centroid, take the same NUMBER of blocks Sol took, and report B(i). Also report the share of the `blockmax` arm's gain over the centroid rule that each one recovers. The same arm feeds the tau-sigma rule unchanged if the ranking is simply swapped.

3. **Items 1 and 2 together: the full 16×16 cell grid with separate per-warp decisions.** [I]
   - **Cost.** 16 G²d = 1/512 of dense. That is still about a fifth of `token_aug`'s cost at u = 0.8 with the tail on, and about a third with it off (§5).
   - **Measure first.** Whether the gains of items 1 and 2 add. If they do not, ship the cheaper of the two.

4. **If `token_aug` is ever turned on: cut its scan, not its budget.** [I; V, evidence]
   - **Why the scan.** The shipped-config comment records that budgets 64, 128 and 256 are indistinguishable in isolated kernel time (`workflows/h3_config.py`, above `token_aug_blocks=""`). So the cost is the scan over every unrouted key, which §5 prices at 3u/256.
   - **Three 2026 changes.**
     - HISA's hierarchy: scan only the top-m unrouted blocks by pooled score. HISA's cost is O(L²/B + L·m·B) against O(L²) (its Eq. 11), and it reports 92.6–94.9% per-layer IoU with the flat indexer (its Table 3).
     - PIVOT-Refine: re-score the candidate set per query sub-block rather than per 128-row centroid.
     - Self-Indexing Attention's "absmax-sign" group query: each coordinate keeps the sign of the member with the largest magnitude, which avoids the cancellation a mean suffers.
   - **Measure first.** On captures, the share of the tokens `token_aug` would admit that fall inside the top-m unrouted blocks ranked by pooled score, for m in {16, 64, 256}. That is HISA's IoU question asked of our centroid. If the share is near 1 at small m, the scan can shrink by roughly u·G/m.

5. **Add the value norm to the block score (Stem's output-aware metric, 2603.06274).** [V for the formula; I for the transfer]
   - **The metric.** Stem scores M = QKᵀ + β·max(0, log‖V_j‖₂). The reason: a block with high affinity and a small value contributes little output.
   - **Cost.** Near zero, because kitchen already builds per-block V sums for the pooled tail.
   - **Measure first.** Rank by centroid score plus β·log of the block's V norm. Report the missed OUTPUT (the norm of the missed Σ p·v), not the missed mass, next to B(i), and relative L2 of the emulated output.

6. **Reuse, to afford an expensive scorer, is not needed at these costs.** [I]
   - The scorers in items 1–3 cost well under 1% of dense (§5).
   - Reuse matters only for scorers near XAttention's 1/16 of dense, or for a full index pass. The peer tool's E rows show routes surviving between steps. DFSAttn refreshes masks every 12 of 50 steps. IndexCache picks which layers keep an indexer by a training-free greedy search.
   - On the 8-step PDD8 schedule there is little to amortise over. The step-stability papers belong to another agent.

7. **Low for H3 now.** [I]
   - Retrieval indexes (PQ, clustering, LSH tables): rebuilt every call here, see the table's facts 1 and 2.
   - 1-bit key sketches: the INT8 per-token score already exists and is exact to INT8; a sketch saves bandwidth, not selection quality.
   - Quest bounds: measured worse locally.
   - Query-aware MLLM pruning: conditioning rows are already exact in Sol.
   - Encoder-side merging: `docs/research/technique_transfer.md`'s open borrow 1 is unchanged by 2026 work.

**The blind spot to keep in view.** Every "measure first" above is a capture proxy. The repo's own evidence says such proxies have anti-predicted pixels once: `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:48-61` shows better on both offline proxies and worse on 0 of 15 prompts. `internal/2026-09-19_question_review.md` C.11 asks for the verdict-separation check before any lane is built on capture error. So treat equal-cost missed mass as a SCREEN that nominates at most one candidate for a blind panel, not as a decision rule.

## 1. Corrections and agreements with existing records

- **[V] Correction to `docs/research/2026-09-17_rotation_and_lowbit_attention_survey.md` §D.9.** That section credits DFSAttn with "layer-wise sparsity profiling + bidirectional co-clustering". The DFSAttn paper (arXiv 2605.23445, "Dynamic Fine-grained Sparse Attention for Efficient Video Generation", 2026-05-22) describes 16-token sub-block pooling, block-score aggregation and 3D Hilbert reordering. The co-clustering mechanism named in D.9's line belongs to arXiv 2603.18636, "Attention Sparsity is Input-Stable ... Offline Sparsity Profiling and Online QK Co-Clustering" (SVOO), which another agent covers. I did not reopen the index D.9 cites, so the cause of the mix-up is not established here. Per CLAUDE.md, log what the line used to claim in `docs/wiki/decisions.md` when it is fixed.
- **[V] Agreement with `docs/research/technique_transfer.md`:**
  - Its row "learned sparse attention with block selection (DeepSeek DSA, NSA, MoBA) → exists" holds. Every 2026 LLM indexer found here is learned too: QSA's `index_qk_proj`, MSA's KL-trained index branch, SpotAttention's KL-trained selector.
  - **This draft goes beyond that row in two ways.** First, the training-free 2026 work sits in the search path around learned indexers, and its ideas (hierarchy, query-group proxies, head routing, cross-layer reuse) do transfer. Second, the table has no row for retrieval-style scoring. Such scoring fails here for the table's own facts 1 and 2, not for a new reason.
  - Its row "token merging and pruning → possible, unmeasured" is unchanged by the 2026 MLLM and encoder work in §4.3–4.4. Trend-aware Pruning's finding that importance moves between layers is one more reason to expect fact 2 to bite.

## 2. Family 1: LLM sparse-attention indexers and training-free prefill scorers (not DeepSeek's own)

Legend: cost is relative to the attention it gates, S is sequence length, d is head_dim.

| method | scorer (what, precision) | granularity q × k | cost | reused across | training-free? | code | measured vs claimed |
|---|---|---|---|---|---|---|---|
| **Qwen QSA (Qwen3.8-Flash-Next / "Qwen4-Exp")** | learned `index_qk_proj` (`coderef/sglang/python/sglang/srt/layers/attention/qsa/qsa_indexer.py:69`); keys mean-pooled over `compress_ratio` tokens in fp32 (`coderef/sglang/python/sglang/srt/layers/attention/qsa/kernel.py:12-20`), RMSNorm then RoPE at the block's position (`coderef/sglang/python/sglang/srt/layers/attention/qsa/qsa_indexer.py:205`); score = Σ over index heads of ReLU(q·k̄)/√d, "weight-free" = no per-head weights (`coderef/sglang/python/sglang/srt/layers/attention/qsa/mqa.py:48-49`) | 1 query token × 4-token micro-block (Raschka's gallery; config requires top-k blocks ∈ {512, 2048}, `coderef/sglang/python/sglang/srt/layers/attention/qsa/config.py:16-17,64-68`); expanded to tokens plus the current partial block (`coderef/sglang/python/sglang/srt/layers/attention/qsa/kernel.py:71-127`) | one index-head GEMM over S/4 pooled keys per query token | shared by all attention heads (one index KV head, `coderef/sglang/python/sglang/srt/layers/attention/qsa/config.py:51-52`); NOT across layers: "QSA compresses independently per layer" (`coderef/sglang/docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx:126`) | no (learned projection) | yes, sglang | the cookbook quotes Qwen's kernel speedups at 1M tokens; not reproduced |
| QSA **tokenwise** form (removed) | per-token index-K cache, "Qwen3Next-DSA" (docstring remnant, `coderef/sglang/python/sglang/srt/mem_cache/qsa_kv_pool.py:3-6`) | token × token | O(S) index scan per query | — | no | removed by sgl-project/sglang PR 38960, "[Qwen 3.8 Next] Remove unused tokenwise QSA implementation and tests", merged 2026-09-12 [V title and date; diff not read, since git is forbidden here and the PR page did not list files] | the difference that matters: compressed QSA pools keys 4:1 BEFORE scoring, so its index scan is 4× cheaper and selection is in whole micro-blocks |
| **MiniMax Sparse Attention (M3)**, arXiv 2606.13392 | learned index branch, one index query head per GQA group and one shared index key head (paper Eq. 5, per the fetch); block score = MAX over the block's exact per-token index logits (Eq. 6); sglang also allows `lse` (`coderef/sglang/python/sglang/srt/configs/model_config.py:340-346`, kernel `coderef/sglang/python/sglang/kernels/ops/attention/minimax_sparse/prefill/flash_with_topk_idx.py:231-239`) | query token × 128-token block, top-16 plus init and local blocks (`coderef/sglang/python/sglang/srt/layers/attention/minimax_sparse_backend.py:162-175,555`) | dense index pass H_kv·d_idx·S² (paper Eq. 12) | shared within a GQA group; per layer | no: KL-trained index branch (paper §3.2, Eq. 10) | yes (sglang, vLLM, flashinfer `msa_ops`) | measured 14.2× prefill, 7.6× decode on H800 at 1M; the 28.4× is a FLOP claim (paper §5.4); the fetch reports no max/mean/LSE ablation |
| flashinfer **MSA ops** | stage 1: per-(head, 128-token KV block, query token) max of raw QKᵀ, no softmax, no V; `reduce_heads` max-reduces to one shared selection ("MiniMax-M3 indexer semantics") (`coderef/flashinfer/flashinfer/msa_ops/proxy_score.py:282-336`; `coderef/flashinfer/flashinfer/msa_ops/_common.py:22` `_BLK_KV = 128`) | token × 128 | full QKᵀ at index width | across heads if `reduce_heads` | kernel only | yes; SM100/103/120/121 only (`coderef/flashinfer/flashinfer/msa_ops/__init__.py:1-5`) | — |
| flashinfer **QToken-KvBlock-Sparse-Attention** (prims-ts) | consumes an indexer's output, scores nothing itself | **per query token** a list of logical K/V block ids; `kv_block_size` ∈ {4,8,16,32,64,128} is the "semantic indexer atom", decoupled from physical `page_size` (`coderef/flashinfer/flashinfer/attention/prims_ts/README.md:62-77`) | executes G queries as one tile over the sorted, unique-reduced UNION of their ≤ G·(block_topk+1) candidates, with per-query membership bits (`coderef/flashinfer/flashinfer/attention/prims_ts/README.md:107-115`) | pattern shared across KV heads by default (`coderef/flashinfer/flashinfer/attention/prims_ts/README.md:65-67`) | — | yes; causal only, SM100a/B200 signed off (`coderef/flashinfer/flashinfer/attention/prims_ts/README.md:13-15,124-128`) | — |
| flashinfer block-sparse FMHA, `use_proxy_routes` | "one K arithmetic mean and one V sum per semantic KV block"; optional `kv_valid_bits` filters exact tokens inside routed blocks (`coderef/flashinfer/flashinfer/attention/prims_ts/README.md:187-208`) | block × block, plus a token bitmask | — | — | — | Blackwell | this is Sol's pooled-tail contract as a library primitive |
| **MoBA** (2502.13189, 2025 base) / VMoBA (2506.23858, 2025, ICLR 2026) | mean-pooled key chunk · each query TOKEN, fp32 (`coderef/sglang/python/sglang/multimodal_gen/csrc/attn/vmoba_attn/vmoba/vmoba.py:657-671`); top-k or cumulative-threshold selection (`coderef/sglang/python/sglang/multimodal_gen/csrc/attn/vmoba_attn/vmoba/vmoba.py:76-147`) | token × chunk (the reverse of `token_aug`) | S·(S/chunk)·d | per head, per layer | MoBA is trained; VMoBA claims a training-free mode (abstract) | yes | VMoBA's abstract reports training speedups |
| NSA (2502.11089, 2025 base) / **InfLLM-v2** (MiniCPM, in sglang) | per query token against compressed keys over overlapping windows (`kernel_size`, `kernel_stride`), max-pooled into blocks, top-k plus init and local blocks (`coderef/sglang/python/sglang/srt/layers/attention/minicpm/sparse_utils.py:216-296`) | token × block | S·(S/stride)·d | per layer | no | yes | not read |
| **HISA**, 2603.28458 (2026-03-30) | stage 1: mean-pooled block key per B = 128, keep top m = 64 blocks; stage 2: the original DSA indexer over those m·B tokens (§4.1, per the fetch) | token × block, then token × token | O(L²/B + L·m·B) vs O(L²) (Eq. 11) | none | yes (on a learned indexer) | github.com/MuLabPKU/TransArch | measured: up to 3.75× indexer speedup at 64K, 1.65× throughput in SGLang (Table 1); IoU with flat DSA 92.58–94.94% by layer, falling from 96.91% to 78.66% by position (Table 3). Stated failure (App. C.5): mean pooling dilutes needles and short spans |
| **PIVOT**, 2607.24593 (2026-07-27) | proxy query = per-head mean of g members' index queries and gating weights; one shared prefix scan gives c = 2k candidates; Refine re-scores the candidates per query | group of g = 4 (prefill) × token | O(L + g·c) per group vs O(g·L) | across the g queries (Reuse variant) | yes (on DSA / GLM-5.1 indexers) | not linked | measured on DeepSeek-V3.2 and GLM-5.1: accuracy matches dense DSA (Table 1); up to 4.8× indexer kernel speedup (Fig. 4a). Their Fig. 2: adjacent queries share about 0.8–0.9 of top-k; the union over g = 4 is 1.3–1.5 k |
| MISA, 2605.07363 (abstract) | a router on cheap block statistics picks a few of the DSA indexer heads per query; a hierarchical variant re-ranks an enlarged candidate set with the full indexer | token | fewer index heads | — | yes | TileLang (abstract) | abstract: matches DSA on LongBench, recovers >92% of DSA's tokens per layer |
| IndexCache, 2603.12201 (abstract) | "Full" layers run the indexer, "Shared" layers reuse the nearest Full layer's top-k; the training-free variant picks the layers by greedy search on calibration LM loss | — | removes 75% of indexer work (abstract) | **across layers** | the greedy variant, yes | — | abstract: 1.82× prefill on a 30B DSA model; "preliminary" on GLM-5 |
| AsyncTLS, 2604.07815 (abstract) | coarse block filter, then fine token selection | block → token | — | temporal locality for offload | not stated in abstract | — | abstract only |
| **XAttention**, 2503.16428 (2025 base) | antidiagonal sums at stride S inside each B×B block, computed as strided-reshaped QKᵀ; softmax; smallest block set reaching cumulative τ (Alg. 1, §2) | block × block | ≈ S²d/stride, i.e. 1/(2·stride) of dense attention [I, from the reshape] | per head (DP-tuned τ) | yes | github.com/mit-han-lab/x-attention | video: HunyuanVideo, dense for the first 5 of 50 steps, stride 8 (Table 4) |
| **Stem**, 2603.06274 (v2 2026-07-31) | output-aware metric QKᵀ + β·max(0, log‖V_j‖), block-downsampled; position-decaying budget for early (causal) tokens | block | O(2N²d/B²) for the metric (per the fetch) | — | yes | Triton said to be open, no link | Tables 1–2 on Llama-3.1-8B and Qwen3-8B |
| Token Sparse Attention, 2602.03216 (ICML 2026) | recent queries' attention summed per key, per head (a causal-LLM proxy); selected in about half the layers by a representation-drift test; dropped tokens re-enter in later layers | token | "<11% of attention latency" at 128K (their claim) | not across layers (re-selected) | yes | github.com/dongwonjo/Token-Sparse-Attention | Table 1, Llama-3.1-8B / Mistral-Nemo |
| ProxyAttn, 2509.24745 (2025) | scores of pooled representative heads stand in for all heads; per-head dynamic budgets | block | head-pooled | **across heads** | yes | github.com/wyxstriker/ProxyAttn | abstract only |
| **CompactAttention**, 2605.16839 (2026-05-16) | takes 2D block masks (SeerAttention, FlashPrefill) and ORs them over query blocks and over heads in a KV group into one KV block table | chunk × block | — | across query blocks and heads (by union) | inherits the scorer | github.com/jiwonsong-dev/CompactAttention | **measured union inflation**: Qwen3-30B-A3B at 128K, sparsity 93.88% before union → 86.63% after the Q-block union → 70.72% after the sub-KV-group union (Table 2); 2.72× attention speedup on H200 |
| SpotAttention, 2606.22874 | learned tiny Q-K scorer, KL-distilled against the frozen backbone, 16-token blocks | block × block | selector 1.55 ms against 2.97 ms of sparse attention at 128K (Table 1, their H/W) | per layer | **no** (trained plug-in) | none linked | decode only; sparse prefill "not benchmarked" (their words) |

- **[?] Kimi.** Search results say Kimi K3 (2026-07) is a Kimi Delta Attention (linear) plus full-attention hybrid. I found no Moonshot 2026 sparse-attention indexer beyond MoBA (2025) and FlashMoBA (2511.11571, 2025). Read as search snippets only.
- **[?] GLM.** GLM-5 / 5.1 / 5.2 use DSA (NVIDIA NeMo docs title). GLM-5.2's "IndexShare" appears only in blog titles I did not open. IndexCache above is the paper-level source.
- **[V] sglang's `nsa/` directory is not NSA.** It is a deprecated shim for DeepSeek's DSA indexer (`coderef/sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py:1`, "use dsa.dsa_indexer instead"), so it is out of scope here.
- **[?] FlexPrefill and SampleAttention** (2502.20766, 2406.15486) are 2024–2025 bases. Both estimate the pattern from a subset of query rows, which is a causal-LLM device. Titles verified only. Not re-read.

**What transfers from family 1 [I].** Four things:
- Shared proxies for a group of queries, followed by per-member refinement (PIVOT, and kitchen's `TOK_GROUP` centroid is the no-refine version).
- A block filter before any token scan (HISA, AsyncTLS).
- Max or LSE over exact sub-scores rather than a mean (MSA, InfLLM-v2).
- Union-of-query-tile execution with membership bits (flashinfer, MSA, CompactAttention).

What does not transfer: the learned projections, and every trick that leans on causality (recent queries as a proxy, position-decay budgets).

## 3. Family 2: retrieval-style scoring of keys

| method | estimator or bound | granularity | index build | failure mode |
|---|---|---|---|---|
| Quest, 2406.10774 (2024 base) | upper bound Σ_c max(q_c·min_c, q_c·max_c) from per-channel min/max per page | query × page | one pass over K | **Measured on H3 captures by the peer tool (arm D):** the bound dominates the true block max as it must, yet ranks worse than Sol's centroid rule on most cells. It rewards a block for being wide at d = 128 over 64 keys (`bench/analyze_sol_block_grouping.py`, `VERDICT`) |
| MagicPIG, 2410.16179 (2024 base) | LSH (SimHash) sampling, importance-weighted estimate of the attention output | token | hash tables built at prefill, CPU-hosted | [?] mechanism details not re-read; anisotropic keys hurt LSH (ParisKV's and a search snippet's framing) |
| PQCache 2407.12820, RetrievalAttention 2409.10516, ClusterKV 2412.03213, Squeezed Attention 2411.09688, Multipole 2506.13059 (2024–25 bases) | PQ codes / ANN graph / semantic clusters / offline key clusters; Multipole also uses centroids to APPROXIMATE the unselected keys | token or cluster | k-means, graph or PQ training, amortised over decode | [V, abstracts] every one amortises an index over many decode queries against a fixed cache. Multipole's "centroids to identify important keys and approximate the rest" is Sol's pooled tail with clusters in place of contiguous blocks |
| **ParisKV**, 2602.07721 (2026-02-07, abstract + code link) | collision-based candidate selection, then a quantized inner-product rerank; random rotation onto a hypersphere; data-independent centroids | token | cheap, data-independent | abstract claims drift-robustness; code github.com/amy-77/ParisKV |
| **RaBitQCache**, 2606.31519 (2026-06-30) | random orthogonal rotation, 1-bit key codes sign(Pᵀk), INT4 query; estimator ⟨c̄, q′⟩/α with α = ⟨c̄, Pᵀk_c⟩, **unbiased with O(1/√D) error w.h.p.** (Thm A.1; INT4 adds O(1/√D), Thm A.2); adaptive **top-p** budget | token | one pass over K | assumes a near-uniform spread on the sphere; decode-only, LLMs to 64K (Tables 2, 4); code github.com/Sakuraaa0/RaBitQCache |
| **Self-Indexing Attention**, 2609.13205 (2026-08-16) | randomized Hadamard D·H per layer and KV head; score = max(agree(q,k) − d/2, 0)·‖k̃‖ from sign agreement; **grouped prefill queries of 64 by "absmax-sign pooling"** | 64-query group × token | one pass; the 1-bit index is shared by prefill and decode | no formal guarantee (their App. B); claims selector ≈ 1/256 of dense QK and 1-bit GEMM ≈ 8× BF16 throughput; code not linked |
| Late interaction (ColBERT MaxSim) | Σ over query tokens of max over key tokens | query block × key block | — | an exact "block MaxSim" is a full score pass; the peer tool's `blockmax` arm is exactly that and is a bound, not a rule. TileMaxSim (2606.26439, abstract) is an IO-aware MaxSim kernel family on H100 with fused PQ scoring. ColBERT-Att (2603.25248): search title only [?] |

**[I] What survives the DiT setting.** Only estimators whose index is one linear pass over the current K qualify: per-block min/max (measured too loose here), sign bits after a rotation, RaBitQ codes and sub-block means.

**On this box [I]:**
- Kitchen's `rotate` option already applies a sign-diagonal Hadamard H128 before the INT8 quantizers (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/__init__.py:2658-2663`). That is Self-Indexing's transform. Its sign bits would be a by-product of a pass the kernel can already make.
- But `token_aug` already scores every candidate token at INT8 precision. A 1-bit sketch would buy bandwidth, not selection quality.
- [?] Whether sm89's b1 tensor-core MMA is usable from this kernel was not checked.
- RaBitQ's top-p rule is the one idea here that bears on the budget rather than the scorer: an unbiased per-token estimate lets the budget follow the mass. That is the per-head budget lever in D.7 by another route.

## 4. Families 3–5

### 4.1 Sub-block routing in video DiTs (family 5), and how kernels keep tiles dense

| method | routes at | computes at | scorer | cost | reuse | training-free | runs on sm89? |
|---|---|---|---|---|---|---|---|
| **sglang SubBlock** (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py`, read in full) | 16 × 16 pooled cells (n_q = n_k = 4 in a 64 × 64 block) | Q64 × K64 (Q64 × K128 for SM90 sage_fp8) | log Σ_{a,b} exp(q̄_a·k̄_b·scale) over the 16 cell pairs, **fp32 scores** (bf16 ties biased selection toward early key blocks, +3.9% rel-L2 at S = 96k, `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:257-260`); uniform top-k, budget snapped to 8 blocks (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:159-173`) | 16 G²d = 1/512 of dense [I, arithmetic]; their "0.5% of denoise time" (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:205`) | per head, per layer and step; no sink or diagonal reservation, measured not to survive to pixels (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:212-215`) | yes | **no**: SM90/100/120 only (prior survey §D.2) |
| **DFSAttn**, 2605.23445 (code read) | pooled tiles: paper 16, public code default 32 (`dfsattn/attention_wan.py:66-67,409`) | 128 × 128 blocks through Block-Sparse-Attention / FA2 | mean-pooled tiles, **softmax over key tiles per query tile, summed per block pair** (`attention_wan.py:150-166,192-195`); top-k by a kept fraction (`:198`) | S²d/tile² per refresh | masks refreshed every `cache_interval` = 12 steps, dense for the first 12 of 50 (`attention_wan.py:27-52,404-405`); per layer and head | yes | the PyTorch scorer yes; its block kernel was not checked |
| **FG-Attn**, 2509.16518 (v1 2025-09, v2 2026-06-04) | M × N with M ≥ 16 queries and N down to 1 key ("M×1 slices") | dense tiles packed by an asynchronous **gather-load** of the selected key columns; query tiles grouped by XOR distance to share loads | mean q/k per slice, top-p 0.9 | their measure: mask 0.06× of dense attention time on average, 0.12× at 16 × 16 | — | yes | built on FA3 for H100; code not linked |
| **kitchen `token_aug`** (local) | 128-query centroid × 1 key | listed keys **gathered into tile layout** for the exact kernel (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_exact.cu:178`) | INT8 centroid·key; histogram (0.25-log2 fine bins), whole-bin admission (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu:17-29,45-55`) | 3u·S²d/128 = 3u/256 of dense with the tail on, 2u/256 off [I] | per head; per 2 query blocks | yes | **yes**; off in graphs and node defaults |
| MOD-DiT, 2601.11641 (abstract) | block | block | a linear model fitted on early-step attention predicts masks for later intervals | sampling-free | across steps | yes | not read |

**[V/I] How the kernels keep tiles efficient when routing is finer than compute.** Three patterns:
- **Union plus membership bits.** Used by flashinfer QToken-KvBlock and MSA. The measured price of the union is in CompactAttention's Table 2 and PIVOT's Fig. 2.
- **Gather-load into dense tiles.** Used by FG-Attn, and already by kitchen's token path on sm89 via cp.async. Kitchen's route and exact kernels are sm_80+ (prior survey §D.2).
- **Reordering so fine patterns become block-contiguous.** Used by DFSAttn's Hilbert curve, and the repo's own Morton/Hilbert reorder (`docs/morton.md`).

Kitchen's exact kernel gives a fourth option cheaply: per-warp (16-row) skipping inside a 64-row CTA. The warps split the query axis (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_exact.cu:40,86`), and the token kernel already skips tiles per warp this way (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu:277-281`). One limit applies on the key side [I]: the INT8 PV MMA consumes keys in 32-key chunks (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh:484`, `TILE_PKC = BLOCK / 32`). So a 16-key skip saves QK work but not PV work unless two adjacent 16-key sub-blocks are both skipped.

### 4.2 Query-aware visual-token pruning in multimodal LLMs, 2026 (family 3)

| method | score | layer | kept set reused later? |
|---|---|---|---|
| HAWK, 2604.07812 | text→vision attention from the FIRST LLM layer's W_q/W_k, **RoPE deliberately omitted**, averaged over all text tokens, weighted by per-head importance found OFFLINE by ablating each head's access to vision (Eqs. 4–6, per the fetch) | layer 1 | yes, the pruned set flows to all later layers (their description; not stated explicitly) |
| SinkPruner, 2609.01004 (abstract) | first drop high-norm outlier tokens (feature and spatial redundancy, attention sinks), then a text-guided pruner | early | — |
| TOPS, 2606.27161 (abstract) | task relevance + coverage + diversity | — | — |
| Trend-aware Pruning, 2607.28341 | per-layer attention scores; "momentum" = differences over a window of 5 layers; re-admits tokens with rising importance | every layer from layer 2 | **no**: re-decided per layer, because importance moves between layers (their premise; no rank-correlation number reported) |

**[I] Transfer to H3 is weak on the selection question.** H3's conditioning rows are already exact in Sol (the sink). The team's question is which VIDEO keys each video query needs. Two lessons do carry:
- SinkPruner's point that score-based selection over-keeps high-norm tokens. This is the block-mean analogue of one loud key dominating a mean, and relevant where K norms are lopsided (`docs/h3_block49_quant_error.md`).
- Trend-aware Pruning's evidence that token importance moves across depth. With fact 2 it argues against reusing a selection across DiT blocks.

HAWK's recipe (a cheap first-layer projection without positional terms, plus offline per-head weights) is the one place where a per-head importance prior is set offline. The node cannot set per head (setting brief), so that transfer is blocked at the kernel.

### 4.3 Encoder-side token reduction (family 4)

- **MaMe/MaRe**, 2604.13432 (abstract): matrix-only merging with an inverse "restoration", applied to SD2.1. **ToMA**, 2509.10918 (2025): GPU-aligned merge for diffusion.
- **LTBM**, 2605.25179 (abstract): audio tokens merged only within a temporal window. Locality helps captioning and is less favourable for multiple-choice audio QA.
- **Stride-k**, 2608.30927 (abstract): keeping every second Whisper token held WER.
- **[I] The selection rules here are similarity, locality and stride. None of them decides what attention computes exactly.** They reduce rows for all later layers, which is `docs/research/technique_transfer.md`'s open borrow 1, unchanged. The one echo for H3's audio span: a local-window pooling cell (16 tokens) is the key-side analogue of LTBM's locality constraint.

## 5. A two-stage design for H3, with costs [I: labelled arithmetic from shapes]

**Shapes.** G = 1,600 key blocks per head, so S = 64·G = 102,400. d = 128, 56 heads, 50 DiT blocks per step. Dense attention per head per call is 2·S²·d MACs (QK plus PV) = 2.68e12. One step is 7.52e15 MACs (56 heads × 50 blocks). Every row counts multiply-accumulates and ignores exp, sort and memory.

| scorer | MACs per head per call | fraction of dense | per step (56 × 50) |
|---|---|---|---|
| Sol route today: INT8 centroid × pooled key, plus pooled PV tail (both N×N centroid quantities, `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_route.cu:17-20`) | 2·G²·d = 6.55e8 | 1/4096 | 1.84e12 |
| + key side 4 sub-means (LSE or max), QK only | 4·G²·d = 1.31e9 | 1/2048 | 3.67e12 |
| + query side split into 4 warps, separate decisions, QK only | 4·G²·d = 1.31e9 | 1/2048 | 3.67e12 |
| same, each warp keeping its own pooled tail (QK + PV) | 8·G²·d = 2.62e9 | 1/1024 | 7.34e12 |
| full 16 × 16 cell grid (SubBlock n_q = n_k = 4) | 16·G²·d = 5.24e9 | 1/512 | 1.47e13 |
| kitchen `token_aug`, 3 passes (2 QK + tail PV, `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu:283-288,338-343`) over all unrouted tokens, u = 0.8, 128 rows per centroid (ragged final group ignored) | 3·u·S²·d/128 = 2.52e10 | 1/107 (1/160 with the tail off) | 7.05e13 |
| `token_aug` restricted HISA-style to m = 64 candidate blocks per centroid | 3·(S/128)·64·64·d = 1.26e9 | 1/2133 | 3.52e12 |
| XAttention-style antidiagonal at stride 8 | S²·d/8 = 1.68e11 | 1/16 | 4.70e14 |
| a dense index pass at full width (MSA- or DSA-shaped, no learned projection) | S²·d = 1.34e12 | 1/2 | 3.76e15 |

**Reading it [I]:**
- **Divide by the routed density to get overhead.** Each "fraction of dense" is the scorer's cost against DENSE attention. Its overhead against Sol's actual exact stage is that fraction divided by the routed density.
- **The fine-grained options are cheap.** Every row down to the 16 × 16 grid sits below `token_aug`.
- **What cannot run every call.** XAttention-class and full-index-pass scorers could run only with cross-step reuse (item 6).
- **Precision and throughput differ.** Kitchen's route is INT8 QK plus bf16 pooled PV. SubBlock computes fp32 scores from bf16 pooled operands. Their relative throughput on sm89 is not measured here.
- **The candidate to take forward.** It is the 4 × 4 separate-decision grid on the existing tau-sigma rule. It would be built on kitchen's warp layout and run as a union walk. It could keep the tail per warp or per block.
- **Where the design stops being cheap.** The cost that is not in the table is union inflation. If each warp's set is small but the four sets are disjoint, the CTA loads four times the tiles. That is why the first measurement must report it.

## 6. The offline test, stated once [I]

On the existing captures (capture sets A/B), in the peer tool's harness and with its limits (fp32, a head prefix of 8 of 56, reordering off), add arms at **equal executed cost**:
- **Sol-64.** The baseline at its shipped tau.
- **`split4`.** Four 16-row sub-blocks, each with its own tau-sigma decision.
- **`lse4` and `max4`.** Key blocks ranked by LSE or max over four 16-key sub-means.
- **`split4 + lse4`.** Both together.
- **`token_aug`-emulated.** Centroid of 128 rows × each unrouted key, whole-bin admission at budget 64.
- **`token_aug`-HISA.** The same, restricted to the top-m blocks.
- **`value-aware`.** Stem's metric at block level.

"Cost" is the number of key TILES the CTA loads (the union, for `split4`) plus the gathered tokens, the unit the kernel pays in. For each arm, report:
1. Missed exact softmax mass per query, as the tool's B(i).
2. Missed output norm.
3. Relative L2 of the emulated output with the pooled tail.
4. The 5th percentile across heads, because D.7 says per-head tails move before means do.
5. Union inflation for `split4`.
6. Route stability across two captured steps, as the tool's E.

**Pass bar.** An arm is nominated only if it beats Sol-64 at equal cost on the mid-depth blocks of both clips. It then goes to one blind pair, and only after C.11's verdict-separation check says the capture metric tracks the owner at all.

## 7. Could not verify

- [?] The diff of sglang PR 38960 (the tokenwise QSA removal). Title and merge date only. I did not run git on the local clone, by rule.
- [?] Qwen's own QSA write-up. Only the sglang cookbook and Raschka's gallery page were read. The gallery gives 7.6× prefill and 4.9× decode at 1M; the cookbook gives 10.2× and 6.6×. They disagree, and neither is our setting.
- [?] Kimi 2026 sparse attention, GLM-5.2 IndexShare, FlexPrefill and SampleAttention mechanisms, MagicPIG details, ColBERT-Att. Search snippets or titles only.
- [?] FG-Attn and Self-Indexing Attention code. Not linked in the pages read.
- [?] Whether sm89's b1 MMA is usable for sign-bit scoring.
- Every paper figure above came through a summarising fetch. Recheck it against the PDF before it enters a record.

## Sources

Local (cited at lines above):
- `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/{sol_attn_token.cu, sol_attn_exact.cu, sol_attn_route.cu, sol_layout.cuh}`
- `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/__init__.py`
- `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py`
- `coderef/sglang/python/sglang/srt/layers/attention/qsa/{config.py, qsa_indexer.py, kernel.py, mqa.py}`
- `coderef/sglang/python/sglang/srt/mem_cache/qsa_kv_pool.py`
- `coderef/sglang/docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx`
- `coderef/sglang/python/sglang/srt/layers/attention/minimax_sparse_backend.py`
- `coderef/sglang/python/sglang/kernels/ops/attention/minimax_sparse/prefill/flash_with_topk_idx.py`
- `coderef/sglang/python/sglang/srt/configs/model_config.py`
- `coderef/sglang/python/sglang/srt/layers/attention/minicpm/sparse_utils.py`
- `coderef/sglang/python/sglang/multimodal_gen/csrc/attn/vmoba_attn/vmoba/vmoba.py`
- `coderef/sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py`
- `coderef/flashinfer/flashinfer/attention/prims_ts/README.md`
- `coderef/flashinfer/flashinfer/msa_ops/{__init__.py, _common.py, proxy_score.py, sparse_prefill.py}`
- `workflows/h3_config.py`
- `bench/analyze_sol_block_grouping.py` (uncommitted peer work)
- `docs/research/technique_transfer.md`
- `docs/research/2026-09-17_rotation_and_lowbit_attention_survey.md`
- `internal/2026-09-19_question_review.md`

Remote code:
- https://github.com/jessica-hujie/DFSAttn (`dfsattn/attention_wan.py`, read)
- https://github.com/sgl-project/sglang/pull/38960 (title and merge date)

Papers (arXiv ids verified 2026-09-19):
- MiniMax Sparse Attention https://arxiv.org/abs/2606.13392
- PIVOT https://arxiv.org/abs/2607.24593
- HISA https://arxiv.org/abs/2603.28458
- MISA https://arxiv.org/abs/2605.07363
- IndexCache https://arxiv.org/abs/2603.12201
- AsyncTLS https://arxiv.org/abs/2604.07815
- Self-Indexing Attention https://arxiv.org/abs/2609.13205
- RaBitQCache https://arxiv.org/abs/2606.31519
- ParisKV https://arxiv.org/abs/2602.07721
- Token Sparse Attention https://arxiv.org/abs/2602.03216
- Stem https://arxiv.org/abs/2603.06274
- SpotAttention https://arxiv.org/abs/2606.22874
- CompactAttention https://arxiv.org/abs/2605.16839
- DFSAttn https://arxiv.org/abs/2605.23445
- FG-Attn https://arxiv.org/abs/2509.16518
- MOD-DiT https://arxiv.org/abs/2601.11641
- SVOO ("Attention Sparsity is Input-Stable", named only for the correction) https://arxiv.org/abs/2603.18636
- HAWK https://arxiv.org/abs/2604.07812
- SinkPruner https://arxiv.org/abs/2609.01004
- TOPS https://arxiv.org/abs/2606.27161
- Trend-aware Pruning https://arxiv.org/abs/2607.28341
- MaMe/MaRe https://arxiv.org/abs/2604.13432
- LTBM https://arxiv.org/abs/2605.25179
- Stride-k https://arxiv.org/abs/2608.30927
- TileMaxSim https://arxiv.org/abs/2606.26439

2024–2025 bases:
- XAttention https://arxiv.org/abs/2503.16428
- MoBA https://arxiv.org/abs/2502.13189
- FlashMoBA https://arxiv.org/abs/2511.11571
- VMoBA https://arxiv.org/abs/2506.23858
- NSA https://arxiv.org/abs/2502.11089
- ProxyAttn https://arxiv.org/abs/2509.24745
- FlexPrefill https://arxiv.org/abs/2502.20766
- SampleAttention https://arxiv.org/abs/2406.15486
- Quest https://arxiv.org/abs/2406.10774
- MagicPIG https://arxiv.org/abs/2410.16179
- PQCache https://arxiv.org/abs/2407.12820
- RetrievalAttention https://arxiv.org/abs/2409.10516
- ClusterKV https://arxiv.org/abs/2412.03213
- Squeezed Attention https://arxiv.org/abs/2411.09688
- Multipole Attention https://arxiv.org/abs/2506.13059
- ToMA https://arxiv.org/abs/2509.10918

Web pages:
- https://sebastianraschka.com/llm-architecture-gallery/qwen-sparse-attention/
