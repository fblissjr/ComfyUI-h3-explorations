# DeepSeek V4.1's hierarchical sparse indexer and layer modes, read for MiniMax H3

> **Provenance, 2026-09-19.** Written by a research subagent of the `libguy`
> session on the lead's brief (item 1 of the second brief), read-only, no GPU.
> The commissioning session re-checked six of its pointers against the
> sources and found them correct: the block-max pool in
> `coderef/sglang/python/sglang/srt/layers/attention/dsv4/candidate_indexer.py:94-118`,
> the below-window collapse in
> `coderef/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py:3372-3378`,
> `TOK_GROUP` in `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh:45`,
> the token-tile gather in
> `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_exact.cu:178-190`,
> the head count in ComfyUI core `comfy/ldm/minimax/model.py:475`, and the
> block-49 `k_norm.weight` note at `docs/roadmap.md:826-835`. The rest is as
> the agent reported it, with its own labels. Figures quoted from the report
> or other repos are theirs, on their setups. `bench/analyze_sol_block_grouping.py`
> was a peer session's uncommitted work when this was written.

Written by a research subagent on 2026-09-19. Read-only: no git, no GPU, no
file in the repo or in `coderef/` touched. Sources reached: the DeepSeek-V4.1
tech report itself (`DeepSeek_V41_Tech_Report.pdf`, 51 pages, downloaded from
the Hugging Face model repo and read as text; page numbers below are the
printed page numbers, which match the PDF page index), the model's
`config.json` from the same repo, the local clones `coderef/sglang`,
`coderef/vllm`, `coderef/flashinfer` and `coderef/comfy-kitchen` as they stand
on disk (commit identity not checked, because git was off-limits), and the
abstracts of IndexCache (arXiv 2603.12201) and HISA (arXiv 2603.28458) plus
quoted sentences from HISA's HTML method section. Not reached: the
DeepSeek-V4 report (which defines CSA and its indexer; the V4.1 report defers
to it), the full text of IndexCache, and the DeepSelect and FlashMLA kernel
sources (not in any clone). Labels: [V] read, with a pointer; [I] inference or
arithmetic from shapes; [?] could not verify.

## What this means for H3, ranked

1. **Token-level selection is cheap in DeepSeek because of MQA, and H3 has no
   MQA.** [V] DeepSeek's main attention has one KV head (`num_kv_heads=1`,
   `coderef/sglang/python/sglang/srt/models/deepseek_v4.py:1169`; `config.json`
   `num_key_value_heads = 1`) and K is V (`assert k is v`,
   `coderef/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py:3730`).
   One index list per query token serves all 64 query heads, and the heads
   fill the MMA's M dimension ("FlashMLA's FP8 decode takes only h_q in {64,
   128}", `coderef/vllm/vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py:80`).
   [I] H3 is MHA with 56 independent K/V heads (ComfyUI core
   `comfy/ldm/minimax/model.py:475` sets `num_attention_heads=56`; `:175`
   splits q, k and v equally), so a gathered key row serves one
   head only. On H3 the amortisation has to come from the query side: one key
   list shared by a whole query block, gathered into a 64-row tile. That is
   exactly what comfy-kitchen's `token_aug` already does (per head, one list
   per `TOK_GROUP = 2` query blocks, gathered by `cp.async` into the block
   layout: `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_exact.cu:178-228`,
   `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh:45`).
   The kernel shape for "finer than 64 on the key side" exists; nothing in
   the LLM clones improves on it for an MHA model.

2. **The hierarchy's motive is absent in Sol unless a token-level pass is
   added.** [I, arithmetic from `config.json` shapes] In V4.1's decoder the
   indexer costs 32 heads x 128 dims per (query, position) and the main
   attention costs 64 heads x 512 dims x 2 per selected entry over 512 + 128
   entries, so the indexer's MACs overtake the attention's past roughly ten
   thousand positions; at long context the indexer is the bottleneck, which
   is the stated reason for 2.3.2 (p. 11). In H3, Sol's block routing costs
   `(T/64)^2 d H` MACs, 1/4096 of dense `QK^T`, and so is negligible next to
   the exact stage. The pool-then-refine idea only pays on H3 if we add a
   per-token score pass, which costs 1/64 of dense `QK^T` (1/128 with
   kitchen's `TOK_GROUP = 2`). `token_aug` already runs that pass, over all
   unrouted tokens.

3. **The training-free form of this idea is HISA, not CSA2, and it is Sol +
   token_aug in a different order.** [V] The report says the hierarchical
   indexer "is training-aware and introduced in post-training" (§2.3.2, p. 12),
   and the whole model trains sparse attention "from scratch ... without any
   dense attention warmup" (p. 6). HISA, which the report cites as the prior
   work (p. 11), "requires no additional training", scores "a representative
   key ... via mean pooling over its indexing keys", keeps the first and last
   blocks always, then runs the original token-level indexer only inside the
   retained blocks. [I] HISA's stage 1 is Sol's block mean; its stage 2 is a
   token pass restricted to a pool. The direct H3 experiment: restrict
   `token_aug`'s token pass from "every unrouted token" to the top-P
   unrouted blocks by block score, which cuts its cost to a fraction `P /
   (unrouted blocks)` (section 3.2). The pool must be skipped, not masked:
   off Blackwell both sglang and vLLM compute full logits and mask them,
   which changes the selection and saves nothing (section 2.2).

4. **Of the three modes, only index Reuse has an H3 analogue, and it is across
   denoising steps, not across blocks.** [I] Full and Reindex share main KV and
   indexer K across layers; that needs trained projections (V4.1 decoder KV is
   projected from the encoder's last hidden state, §2.2 p. 9) and it is
   blocked on H3 by `docs/research/technique_transfer.md` fact 2 (every row
   updated every block), which agrees with that table's "KV cache
   quantisation: no KV survives a step: n/a" row. Reusing *indices* is
   different: they are routing metadata, not KV, so fact 2 does not forbid
   it; whether they are stable is the empirical question. That goes beyond
   the table's rows. The fourth axis of the in-flight, uncommitted
   `bench/analyze_sol_block_grouping.py` ("if the routed set barely moves
   between denoising steps, an expensive decision can be taken once and
   reused") is this question already framed; read its record,
   `bench/results/2026-09-19_sol_block_grouping.json`, before designing
   anything here. The number to beat: DeepSeek pays one full-range scoring
   pass and spreads it over four re-indexing layers in one forward (layers
   24, 28, 32, 36, section 1.2). The H3 analogue is how many steps a route
   survives.

5. **Which layers stay Full: DeepSeek states a schedule and does not justify
   it; IndexCache gives a training-free method.** [V] The report fixes the
   schedule in §4.2.1 (p. 22) with no search or ablation. IndexCache chooses
   the layers that keep their own indexer by "a greedy search algorithm ...
   directly minimizing language modeling loss on a calibration set, requiring
   no weight updates" and reports removing 75% of indexer computations on a
   30B DSA model (abstract). [I] On H3 the equivalent search would pick
   which (step, block) pairs recompute routes, graded on captured
   activations first, which is this repo's standing rule for numerical
   changes.

None of the DSv4.1, MSA or prims_ts kernels in the LLM clones runs on sm89 (section 2.4). Four points in the
relayed description need correcting (section 5).

## 1. What the report says

### 1.1 The three modes [V, §2.3 p. 10-11, Fig. 4 p. 10]

- "Each CSA2 layer is statically assigned one of three modes: Full, Reindex,
  or Reuse. In all three modes, the layer computes its own query and SWA KV"
  (§2.3.1, p. 10).
- Full: "computes its own main KV and indexer Q, projects indexer K from that
  main KV, and runs the indexer to produce fresh Top-K indices" (p. 10).
- Reindex: "reuses the most recent available main KV from a preceding layer
  together with its corresponding indexer K. The indexer computes its own
  query, rescores the reused keys, and produces fresh Top-K indices" (p. 11).
- Reuse: "reuses the most recent available main KV and the latest Top-K
  indices computed against that main KV by a preceding layer in Full or
  Reindex Mode ... without computing indexer Q or evaluating index scores"
  (p. 11).
- Two simplifications against V4's CSA: no overlap between compressed
  entries and no absolute positional embedding in the compressor, and
  indexer K is a projection of main KV rather than a separate compression
  path (§2.3, p. 10).

### 1.2 Which layers get which mode, and how chosen [V]

- §4.2.1 (p. 22): layers 0-1 pure sliding window; the other 18 encoder layers
  are CSA2 at compression 2 in "three identically configured groups of six
  layers", Full then five Reuse; the 20 decoder layers are CSA2 at
  compression 1 in five groups of four, the first group Full then three
  Reuse, the other four groups Reindex then three Reuse.
- `config.json` agrees: `kv_source_layer_ids = [2, 8, 14, 20]` (Full),
  `index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]` (Full plus
  Reindex), `candidate_source_layer_id = 20`, `candidate_topk_blocks = 2048`,
  `candidate_block_size = 8`, `index_topk = 512`, `index_n_heads = 32`,
  `index_head_dim = 128`, `sliding_window = 128`.
- The rule both serving stacks implement: Full is `layer_id in
  kv_source_layer_ids`; Reindex is in `index_source_layer_ids` and not in
  `kv_source_layer_ids`; Reuse is in neither. Against `config.json` that is
  Full {2, 8, 14, 20}, Reindex {24, 28, 32, 36}, Reuse all remaining layers
  from 3 to 39, and layers 0-1 have `compress_ratios` 0 (sliding window only),
  which reproduces §4.2.1's prose exactly.
- Consequences the prose does not spell out: the encoder has no Reindex
  layer at all, and the candidate pool has exactly four consumers (layers
  24, 28, 32, 36) in one forward.
- How chosen: the report says "statically assigned" (p. 4, p. 10) and gives
  no search, sweep or ablation of the schedule. It is stated, not justified.
  The modes are trained with the model (§3.1.2, p. 17-18, is the pipeline
  machinery for sharing across stages).

### 1.3 The hierarchical sparse indexer [V, §2.3.2 p. 11-12, Fig. 5 p. 11]

- Decoder only: "used only in the decoder of CED to reduce this repeated
  scoring during decode" (p. 11).
- "This first Full Mode layer scores all causally visible main KV positions
  and produces the Top-K indices for its own attention. It also performs
  blockwise candidate selection: each block is assigned the maximum index
  score among its positions, and the blocks with the highest scores are
  selected ... For example, selecting 2,048 blocks with 8 positions each
  yields 16,384 candidate positions" (p. 12). §4.2.1 (p. 22) makes 2,048 a
  maximum.
- Reindex layers "score only the candidate positions for the corresponding
  query and select their own Top-K entries within that pool"; Reuse layers
  do no indexing (p. 12).
- The pool is per query and needs "no extra state" because the Full layer's
  scores already exist (p. 11).
- Training-aware: "introduced in post-training: the candidate restriction is
  applied identically during training and inference" (p. 12).

### 1.4 The indexer score [V from code; the V4.1 report defers to V4's CSA]

- The V4.1 report says only that CSA2 "includes a lightweight indexer that
  scores the main KV entries using indexer Q and indexer K" (p. 10). The
  formula is in the code: `s = sum_h w_h(x) * ReLU(q_h . k)`, one K per
  position shared by all 32 indexer heads, head weights from a learned
  `weights_proj` scaled by `index_head_dim^-0.5 * n_heads^-0.5`
  (`coderef/sglang/python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py:253-266`).
  So it is DSA's ReLU-weighted lightning indexer. `wq_b`, `weights_proj`,
  `wk` and `k_norm` ship with the checkpoint (same file, 195-219).
- Precision: indexer Q and K are fake-quantised to FP4 after RoPE with "per-32
  UE8M0 scales" (MXFP4), ties to even
  (`coderef/sglang/python/sglang/kernels/ops/attention/dsv4/torch_quant.py:1-5,44-49`;
  `coderef/sglang/python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py:25-33,234-242`). The report: "We adopt the OCP-standard
  MXFP4 format ... to support as many hardware platforms as possible, despite
  the higher accuracy of alternative formats in our experiments" (§2.4.4,
  p. 14).

### 1.5 What the report measures, and what it only claims [V]

- Measured or derived figures touching CSA2: global KV of "890 bytes per
  token" (p. 1, p. 37), a derived storage count; Reuse layers run "only 15
  kernels during prefill and 11 during decode" (p. 6, p. 19), a kernel
  count; Figure 2's decode FLOPs, a computed count; Table 1 (p. 24),
  end-to-end base-model benchmarks with CSA2 in place.
- Not measured anywhere in the report: recall of the pool, indexer speedup,
  a pool-size sweep, an ablation of the mode schedule, or any per-mode
  quality figure.
- Qualitative claims without numbers: FP4 global KV "with only marginal
  performance degradation" (p. 4); omitting the second-level scale "causes
  no measurable decrease in accuracy" and quantising before RoPE "yields only
  a marginal accuracy improvement" (p. 14); "Potential selection errors in
  CSA2 ... may still cause capability degradation in untested boundary
  cases" (§6, p. 37).

## 2. What the code does

### 2.1 Per question, per clone [V]

| question | sglang | vLLM | flashinfer |
|---|---|---|---|
| (a) indexer score, precision | torch reference `coderef/sglang/python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py:260-266`; SM100 DeepGEMM `fp8_fp4_mqa_logits` on packed E2M1 + UE8M0 (`coderef/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py:316-328`); index-K slot is 64 packed bytes + 4 scale bytes (`coderef/sglang/python/sglang/kernels/ops/attention/dsv4/fp4_indexer.py:13-15`); Hopper decode via a Triton FP4 kernel `fp4_index_logits_decode` (`coderef/sglang/python/sglang/kernels/ops/attention/dsv4/fp4_indexer.py:488-499`) | DeepGEMM `fp8_fp4_(paged_)sparse_mqa_logits`, "DeepGEMM >= 2.8, SM100 only", MXFP4 index cache only (`coderef/vllm/vllm/model_executor/kernels/attention/dsa/sparse_mqa_logits.py:1-19`) | not the DSv4.1 indexer; see MSA below |
| (b) top-k, ties, k | radix top-k, exact k-th value, "which of the elements equal to it fill the last slots is arbitrary", `-1` fill past `min(k, len)`, rows up to 16,384, k up to 2,048 (`coderef/sglang/python/sglang/kernels/ops/attention/dsv4/topk.py:78-99`); indices then sorted ascending (`coderef/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py:3301`) | DeepSelect on the SM100 family; the block top-k uses `torch.topk`, "Keep the existing top-k tie behavior" (`coderef/vllm/vllm/model_executor/kernels/attention/dsa/candidate_blocks.py:179`) | MSA `msa_topk_select`: count-rank or radix, "identical selections only on distinct-score inputs (ties may differ)" (`coderef/flashinfer/flashinfer/msa_ops/sparse_topk_select.py`, `_get_compiled_topk` docstring) |
| (c) block-max pool | **implemented.** torch: pad, `amax` over 8-wide blocks, the block holding the query's newest position forced to `+inf`, `topk` over blocks, mask expanded (`coderef/sglang/python/sglang/srt/layers/attention/dsv4/candidate_indexer.py:94-118`); CUDA `coderef/sglang/python/sglang/kernels/jit/csrc/deepseek_v4/block_amax.cuh` (`kBlockTokens = 8`) and `amax_topk_blocks` (`coderef/sglang/python/sglang/srt/layers/attention/dsv4/candidate_indexer_deep_gemm.py:26,47-70`) | **implemented** in Triton: `_block_scores_kernel` (max with NaN propagation, newest block `+inf`) then `topk` (`coderef/vllm/vllm/model_executor/kernels/attention/dsa/candidate_blocks.py:14-49,141-190`) | not for DSv4.1; MSA's proxy computes a per-128-token-block max (below) |
| (d) mode dispatch | compressor only on `kv_source_layer_ids`, indexer only on `index_source_layer_ids` (`coderef/sglang/python/sglang/srt/models/deepseek_v4.py:1147-1163`); pool flags `is_candidate_source = layer_id == candidate_source_layer_id`, `uses_candidates = 0 <= source < layer_id` (`coderef/sglang/python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py:189-193`) | each consumer resolves its source as `max(s for s in sources if s <= layer_id)` (`coderef/vllm/vllm/models/deepseek_v41/attention.py:291-339`) | n/a |
| (e) gather for main attention | decode: FlashMLA sparse decode reads page-slot index lists into the paged FP4/FP8 caches, list length a multiple of 64 (`coderef/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py:3744-3819`); prefill: dequantise the request's compressed region into a contiguous bf16 workspace `(total_workspace_tokens, 512)`, per-query rebased indices, `flash_mla_sparse_fwd`, top-k padded to 128 (`coderef/sglang/python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py:1-31,45-47`) | FlashMLA sparse (Hopper and Blackwell DC only, `coderef/vllm/vllm/v1/attention/ops/flashmla.py:61-73`); V4.1 mega-attention kernel sm_10x only (`coderef/vllm/vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py:89-100`); CuTe DSL dequant-gather of K into a contiguous buffer (`coderef/vllm/vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py`) | DS4.1 decode: index lists into FP4 global and MXFP8 SWA page pools, "Multiplications use BF16 Q/K/V/P" (`coderef/flashinfer/flashinfer/deepseek_v41.py:13-25`) |

### 2.2 Two implementation facts that change the reading [V]

- **Masking is not skipping.** On Hopper decode and in the dense prefill
  path, sglang computes the consumer layer's full logits and then
  `masked_fill`s everything outside the pool to `-inf` before top-k
  (`coderef/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py:3313-3350,3519-3585`); vLLM's `apply_candidate_mask`
  does the same (`coderef/vllm/vllm/model_executor/kernels/attention/dsa/candidate_blocks.py:192` onward). Only the SM100 decode path
  scores the candidates alone, through DeepGEMM's paged sparse MQA logits
  (`coderef/sglang/python/sglang/srt/layers/attention/dsv4/candidate_indexer_deep_gemm.py:99-119,243-255`). [I] So off Blackwell the
  pool changes *which* tokens are chosen (the model was trained with it) but
  saves no indexer work. The same holds for any H3 version: a pool saves
  compute only if the scoring kernel skips non-candidates.
- **The pool is a no-op below 16,384 positions.** "Every reachable block is a
  candidate inside the window, so the two-level selection collapses to the
  plain top-k" (`coderef/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py:3372-3378`). And the newest block is
  always kept, which the report does not mention (`coderef/sglang/python/sglang/srt/layers/attention/dsv4/candidate_indexer.py:100-112`).

### 2.3 flashinfer items [V]

- **QToken-KvBlock-Sparse-Attention (prims_ts).** Per query token, a list of
  KV-block IDs (block size 4 to 128 tokens) straight from an indexer; a group
  of G query tokens takes the union of its blocks, "sorts and unique-reduces
  them in one CTA, and ORs per-query membership bits"
  (`coderef/flashinfer/flashinfer/attention/prims_ts/README.md:62-76,107-115`).
  Membership is one byte per fragment, so G is at most 8
  (`coderef/flashinfer/include/flashinfer/attention/prims_ts/q_token_kv_block_sparse_metadata.cuh:67`;
  README:126 "Q1--Q8"), and the M dimension is filled by GQA heads:
  "requires `seq_len_q * (Hq / Hkv) <= TileQ128`". It "is causal and
  non-windowed" (README:125-128). Blackwell only: `(10,0), (10,3), (10,7)`
  (`coderef/flashinfer/flashinfer/attention/prims_ts/context.py:65`), validated
  on SM103 (README:15). [I] It is "fine query, coarse key", but the fine query
  is paid for by head packing. For H3 (MHA, `Hq/Hkv = 1`, non-causal) the
  tile would be at most 8 rows tall. What transfers is the pattern: union of
  sub-group selections plus a membership mask inside one tile.
- **Proxy routes in prims_ts block-sparse FMHA.** "A proxy run supplies one K
  arithmetic mean and one V sum per semantic KV block" (README:186-206). [I]
  This is Sol's pooled tail term, in a Blackwell kernel.
- **MSA (MiniMax Sparse Attention, the MiniMax-M3 indexer).** The proxy pass
  computes "the maximum of the unscaled, causally-masked `Q K^T` logits over
  the 128 tokens of KV block `t`, for every query token and query head", with
  an option to max-reduce over proxy heads "(MiniMax-M3 indexer semantics)"
  (`coderef/flashinfer/flashinfer/msa_ops/proxy_score.py:282-291,330-337`).
  SM100/103/120/121, proxy scoring SM120/121 only
  (`coderef/flashinfer/flashinfer/msa_ops/__init__.py:1-5`). A second
  production block-max scorer, on learned proxy projections, dense over
  tokens.
- **SM120 Sage block-sparse** (`coderef/flashinfer/flashinfer/cute_dsl/sparse/bsa_attn_sm120.py:158,373-380`):
  64-token blocks, a `q2k_block_index` per (batch, head, query block), INT8
  Q/K and FP8 V, MHA, head dim 128, non-causal. The closest thing to Sol's
  exact stage in the clones, and SM120a only.

### 2.4 Architectures, and sm89 [V]

- sglang: the candidate indexer returns `None` below sm100
  (`coderef/sglang/python/sglang/srt/layers/attention/dsv4/candidate_indexer.py:42-62`); the V4.1 cache layouts "exist only in SM100
  / SM103 FlashMLA" (`coderef/sglang/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:124-127`);
  the fused FP4 compress kernels need `cvt.rn.satfinite.e2m1x2`, "an sm100+
  instruction" (`coderef/sglang/python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py:89-95`).
- vLLM: FlashMLA sparse is Hopper and Blackwell DC only; DeepSelect and the
  sparse MQA logits path are SM100 family.
- flashinfer: prims_ts SM100/103/107; MSA SM100/103/120/121; SM120 Sage BSA
  SM120a; DS4.1 decode SM100.
- [I] Arch-independent pieces worth reading as references, not running: the
  torch two-level selection in `_low_ratio_index_topk_sm90_decode`
  (`coderef/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py:3519-3585`, which calls the Triton FP4 decode
  kernel), vLLM's Triton block-max kernels, and sglang's Triton tiled sparse
  decode for SM120 (`coderef/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120_triton.py:1-18`),
  a 2-D index gather plus vector mul-reduce, not tensor-core MMA. [?] Whether
  any of the Triton pieces compiles on sm89 was not tried (no GPU).

## 3. Transfer to H3, training-free [I throughout]

Shapes: about 1,600 key blocks per head, so `T` is about 102,400 tokens;
`d = 128`; `H = 56`; 50 blocks. Unit: dense `QK^T` for one DiT block,
`T^2 d H` MACs. All figures below are ratios to that unit, from shapes only.

### 3.1 The closest stand-in for a learned indexer

| scorer | MACs | ratio to dense `QK^T` |
|---|---|---|
| Sol block routing (query centroid x key centroid) | `N_b^2 d H` | 1/4096 |
| per-channel min/max bound per key block (Quest-style upper bound on the block max of a centroid's scores) | `2 N_b^2 d H` | 1/2048 |
| one centroid per query block x every key token (the `token_aug` pass shape) | `N_b T d H` | 1/64; 1/128 at `TOK_GROUP = 2` |
| the same, restricted to a pool of P key blocks per query block | `N_b 64P d H` | `(P / N_b) / 64` |
| token x token on H3's own q, k (DSA-shaped) | `T^2 d H` | 1 |

- A DSA-shaped token-by-token indexer on H3's own q and k costs as much as
  sage's `QK^T`: there is no learned low-dimensional projection to make it
  cheap, and a random or PCA projection is a new approximation with its own
  error to grade. The stand-in has to be coarse on the query side.
- The cheapest honest stand-in is the one already in the kernel: the INT8
  query-block centroid scored against INT8 key tokens, which `token_aug`
  computes with the exact kernel's tile body
  (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu:17-28`).
- Against the exact stage at routed density `rho`, which costs `2 rho` units
  (QK and PV), the token pass over all keys is `(1/64) / (2 rho) = 1 / (128
  rho)` of it with one centroid per query block, and `1 / (256 rho)` at
  kitchen's `TOK_GROUP = 2`. At the 19.65% effective density Sana recorded at
  38k tokens (survey D.5) that is about 4% and 2%, before any pool, and it
  is the ceiling on what a pool can save. Treat the density as illustrative; H3 at 104k has not
  been measured here.
  *Corrected the same day: this counts one QK pass over every key. The
  kernel makes two QK passes over the unrouted keys (a histogram, then the
  listing) and folds the pooled tail's PV into the second, so its cost is
  three passes over the unrouted share, not one over all keys; the priced
  table is section 5 of
  [`2026-09-19_token_selection_scorers.md`](2026-09-19_token_selection_scorers.md).
  The direction of this paragraph's conclusion (a pool cuts the token pass
  and nothing else in Sol) is unchanged.*

### 3.2 Block-max pool then token-level selection, mapped onto Sol

- **Mean versus max.** By linearity, `qbar . kbar_j` equals the mean over
  the block's tokens of `qbar . k_s`. Sol's routing score is therefore
  already the block *mean* of per-token centroid scores, for free. The block
  *max* needs every per-token score: the 1/64 pass. DeepSeek gets the max
  free only because its Full layer scores every position anyway for its own
  top-512. A per-channel min/max bound gives an upper bound on the max at
  1/2048, but it is only a bound.
- **Where the scores already exist.** `token_aug`'s pass 1 computes the
  centroid's score for every token in the unrouted blocks (histogram, then a
  whole-bin threshold). A per-block max over those scores is one extra
  reduction. It could promote an unrouted block whose best token scores high
  into whole-block routing, or seed a pool for later use.
- **The pool as a cost cut for `token_aug`.** Today the token pass covers all
  unrouted tokens. HISA's order would score blocks first (Sol's mean, or the
  min/max bound), keep the top-P unrouted blocks, and run the token pass only
  inside them. That cuts the pass to a fraction `P` over the number of
  unrouted blocks, provided the kernel skips non-candidates rather than
  masking them (section 2.2), and it is the one place in Sol where a
  hierarchy removes real work.
- **Granularity constraint.** The pool atom in every shipped kernel is 8
  positions: `CANDIDATE_BLOCK_SIZE = 8`, "DeepGEMM accepts 8 or 16"
  (`coderef/sglang/python/sglang/srt/layers/attention/dsv4/candidate_indexer_deep_gemm.py:26`); vLLM `SPARSE_BLOCK_KV_CHOICES = (16, 8)`
  (`coderef/vllm/vllm/model_executor/kernels/attention/dsa/sparse_mqa_logits.py:40`). A 64-token pool atom is a new kernel, not a
  config change. Kitchen's token tiles are already the right unit on the
  consumer side: 64 gathered tokens per tile.
- **Finer query side.** The reverse direction (fine query, coarse key) is
  QToken-KvBlock's union-plus-membership pattern. On H3 it would mean
  routing per 16-row sub-block (four centroids per block, 1/1024 of dense
  `QK^T`), taking the union for the 64-row exact tile, masking per
  sub-block, and giving each sub-block its own pooled tail for the blocks it
  did not route. The exact-stage MMA work grows to the union, so this buys
  accuracy at a fixed union, not speed. Whether the query side is where
  accuracy is lost is the question `bench/analyze_sol_block_grouping.py`
  asks; sglang's SubBlock router notes (survey D.3, D.5) are the prior
  evidence.

### 3.3 The three modes, across blocks and across steps

| mode | legs | across H3 blocks in one step | across denoising steps |
|---|---|---|---|
| Full | own KV, own indexer K and Q, fresh top-K | what every block does today | what every step does today |
| Reindex | shared KV and indexer K, own indexer Q | blocked: each block has its own frozen K projection, and K changes every block (fact 2) | blocked for the same reason (K changes every step); the part that survives is "rescore inside an inherited candidate pool" with the current block's own centroids |
| Reuse | shared KV, shared top-K, no indexing | reuse a route computed at block b for blocks b+1.. . Different q/k projections per block make stability doubtful; unmeasured | reuse a (step, block) route at later steps for the same block. Same projections, slowly moving inputs; the most plausible analogue; unmeasured |

- The structural difference: an LLM layer sees a growing causal KV that does
  not change once written, so an index computed at layer 20 stays valid for
  the tokens it names. A DiT block sees the whole sequence rewritten every
  step and every block. Reused indices can only be *stale*, never invalid,
  so the risk is quality, not correctness.
- What a Reuse schedule saves on H3: Sol's block routing is 1/4096 of dense
  `QK^T`, so reusing it saves almost nothing. Reuse pays only if what is
  reused is the expensive part, the token pass (1/64) or a pool derived
  from it.

### 3.4 Which layers must stay Full

- DeepSeek: the first CSA2 layer of every group is Full or Reindex, layers
  0-1 are pure sliding window, and the one decoder Full layer publishes the
  pool. Stated, not justified (section 1.2).
- IndexCache: training-free greedy search on calibration loss (abstract).
  The H3 translation is a greedy search over (step, block) pairs for where
  to recompute routes, graded on captured activations before any render.
- Existing H3-side guidance: the first fifth of the schedule runs dense in
  every shipped config, and sglang measured the dense-*layers* knob as
  nearly free to drop at 38k tokens (survey D.6). Nothing speaks to which
  blocks must recompute a reused route.

### 3.5 Against `docs/research/technique_transfer.md`

- Agrees with the "learned sparse attention with block selection (DeepSeek
  DSA, NSA, MoBA)" row: Sol is the existing analogue, and the learned
  indexer itself does not transfer.
- Agrees with "KV cache quantisation: n/a" and "NVFP4: n/a on this box":
  CSA2's KV and indexer-K sharing and its FP4 cache are KV-cache techniques
  (section 4).
- Goes beyond the table in one place: cross-step reuse of routing *indices*.
  It is not KV reuse, so fact 2 does not rule it out, and it is not the
  declined step caching (that reuses block residuals and skips the block).
  The table has no row for it.

## 4. Item 5: FP4 main KV, FP8 sliding window

- [V] Format: "E2M1 with one E4M3 scale per 16 channels, following NVFP4 ...
  but omitting its second-level global scale"; "We quantize the cache after
  RoPE"; "We retain FP8 for the SWA KV cache due to its sensitivity to
  quantization" (§2.4.4, p. 14). The dynamic-range argument: trained
  RMSNorm weights are about 1, so channel magnitudes are bounded by
  `sqrt(512)` (p. 14).
- [V] It is storage, not compute: FP4 main KV "reduces storage rather than
  accelerates matrix multiplication. Dequantizing cached values before
  attention allows us to use a more accurate format without requiring
  native matrix-multiplication support" (p. 14). flashinfer's DS4.1 decode
  confirms "Multiplications use BF16 Q/K/V/P" (`coderef/flashinfer/flashinfer/deepseek_v41.py:25`).
- [V] Code layouts (`coderef/sglang/python/sglang/kernels/ops/attention/dsv4/kv_layout.py:16-22`):
  `V41` is 512 E4M3 plus 16 UE8M0 scales, one per 32 values, RoPE dims
  quantised too (MXFP8, the SWA cache); `V41_FP4` is 512 E2M1 packed plus 32
  E4M3 scales, one per 16 values (the global cache). Reference quantisers at
  `coderef/sglang/python/sglang/kernels/ops/attention/dsv4/torch_quant.py:52-64,84-103`. flashinfer's writer names them `main_kv_fp4`
  and `swa_mxfp8` (`coderef/flashinfer/flashinfer/deepseek_v41.py:62-63`). Kernels: FlashMLA V4.1 layouts
  SM100/SM103, flashinfer DS4.1 SM100, MSA NVFP4 paged KV SM100/103. None on
  sm89.
- [I] For H3 the storage half is n/a (no KV survives a step, per the table).
  What carries over is the scale *granularity* as an in-kernel Q/K choice,
  which extends survey §C.1 rather than contradicting it: §C.1's "no" for
  per-16 E4M3 micro-scaling is about FP4 compute; DeepSeek shows such a
  format can be dequantised in software, but for H3 that saves nothing,
  because there is nothing to store.
- [I] Nearest sm89 in-kernel alternatives, cheapest first:
  1. Per-channel smoothing between q and k, exact for `q . k` and free in the
     MMA: kitchen's `qk_balance`
     (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/__init__.py:2653-2656`)
     and the planned fold into `k_norm.weight` (`docs/roadmap.md`, the
     2026-09-14 block-49 entry).
  2. Per-channel E4M3 for Q and K on Ada's FP8 tensor cores (survey §C.4).
  3. Per-token, per-32-channel INT8 or E4M3 scales (the MXFP8 shape). The
     scale varies along the reduction axis, and Ada has no block-scaled MMA,
     so the 128-channel dot product becomes four 32-channel partial sums
     (one `m16n8k32` k-step each), each rescaled by `s_q[i,g] * s_k[j,g]`
     in FP32: four times the dequant epilogue of per-row scaling.
- [I] The dynamic-range argument does not transfer. H3's frozen
  `k_norm.weight` is far from uniform at block 49 (`docs/roadmap.md:826-835`
  carries the figures), which is why that block has loud K channels. A
  scheme justified by "norm weights are about 1" needs checking per block
  against `k_norm.weight`.

## 5. Corrections to the relayed description [V]

1. **Two FP4 formats, not one.** The indexer Q/K are MXFP4 (E2M1, one UE8M0
   power-of-two scale per 32 channels, QAT from V4); only the main KV uses
   E2M1 with one E4M3 scale per 16 (p. 14; `coderef/sglang/python/sglang/kernels/ops/attention/dsv4/torch_quant.py:1-5`).
2. **"FP8 sliding window" is MXFP8 in code:** E4M3 with one UE8M0 scale per
   32 values, RoPE dims included (`coderef/sglang/python/sglang/kernels/ops/attention/dsv4/kv_layout.py:19-20`). The report just says
   FP8.
3. **The pool is decoder-only, with one publisher and four consumers.** Only
   decoder layer 20 builds it; encoder Full layers (2, 8, 14) do not, and
   the encoder has no Reindex layers. The 2,048 blocks are a maximum
   (p. 22). The block holding the query's newest position is always kept
   (code, not report), and below 16,384 positions the pool keeps everything.
4. **The mechanism is trained, and so is the schedule.** "Training-aware and
   introduced in post-training" (p. 12); the mode schedule is fixed by hand
   with no reported ablation. Nothing in the report measures the pool or
   the modes in isolation (section 1.5).

The mode definitions, the 2,048 x 8 = 16,384 example, the per-layer top-512,
E2M1 with E4M3 per 16 after RoPE, and FP8 for the sliding window all match
the report as relayed.

## Sources

- DeepSeek-V4.1 tech report, `DeepSeek_V41_Tech_Report.pdf`:
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/DeepSeek_V41_Tech_Report.pdf
  (read: §2.1-2.4.4, §3.1.2, §3.2, §4.2.1, §6, references).
- Model config: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/config.json
- IndexCache, arXiv 2603.12201 (abstract): https://arxiv.org/abs/2603.12201
- HISA, arXiv 2603.28458 (abstract, and quoted sentences from §4.1 and §5.1
  of the HTML): https://arxiv.org/abs/2603.28458, https://arxiv.org/html/2603.28458
- Not read: DeepSeek-V4 report (CSA, lightning indexer definition); YOIO,
  arXiv 2606.06467; DeepSelect (https://github.com/deepseek-ai/DeepSelect);
  FlashMLA sources.
- sglang: `coderef/sglang/python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py`,
  `coderef/sglang/python/sglang/srt/layers/attention/dsv4/candidate_indexer.py`,
  `coderef/sglang/python/sglang/srt/layers/attention/dsv4/candidate_indexer_deep_gemm.py`,
  `coderef/sglang/python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py`,
  `coderef/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py`,
  `coderef/sglang/python/sglang/srt/models/deepseek_v4.py`,
  `coderef/sglang/python/sglang/srt/configs/deepseek_v41.py`,
  `coderef/sglang/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`,
  `coderef/sglang/python/sglang/kernels/ops/attention/dsv4/torch_quant.py`,
  `coderef/sglang/python/sglang/kernels/ops/attention/dsv4/kv_layout.py`,
  `coderef/sglang/python/sglang/kernels/ops/attention/dsv4/topk.py`,
  `coderef/sglang/python/sglang/kernels/ops/attention/dsv4/fp4_indexer.py`,
  `coderef/sglang/python/sglang/kernels/jit/csrc/deepseek_v4/block_amax.cuh`,
  `coderef/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120_triton.py`.
- vLLM: `coderef/vllm/vllm/models/deepseek_v41/attention.py`,
  `coderef/vllm/vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py`,
  `coderef/vllm/vllm/model_executor/kernels/attention/dsa/candidate_blocks.py`,
  `coderef/vllm/vllm/model_executor/kernels/attention/dsa/sparse_mqa_logits.py`,
  `coderef/vllm/vllm/model_executor/layers/sparse_attn_indexer.py`,
  `coderef/vllm/vllm/v1/attention/ops/flashmla.py`,
  `coderef/vllm/vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py`.
- flashinfer: `coderef/flashinfer/flashinfer/attention/prims_ts/README.md`,
  `coderef/flashinfer/flashinfer/attention/prims_ts/context.py`,
  `coderef/flashinfer/include/flashinfer/attention/prims_ts/q_token_kv_block_sparse_metadata.cuh`,
  `coderef/flashinfer/flashinfer/msa_ops/__init__.py`,
  `coderef/flashinfer/flashinfer/msa_ops/proxy_score.py`,
  `coderef/flashinfer/flashinfer/msa_ops/sparse_topk_select.py`,
  `coderef/flashinfer/flashinfer/cute_dsl/sparse/bsa_attn_sm120.py`,
  `coderef/flashinfer/flashinfer/deepseek_v41.py`.
- comfy-kitchen: `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu`,
  `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_exact.cu`,
  `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh`,
  `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/__init__.py`.
- This repo: `docs/research/technique_transfer.md`,
  `docs/research/2026-09-17_rotation_and_lowbit_attention_survey.md` (§C.1,
  §C.4, §D.3, §D.5, §D.6), `docs/roadmap.md`,
  `bench/analyze_sol_block_grouping.py` (uncommitted peer work, read for its
  framing only). ComfyUI core `comfy/ldm/minimax/model.py`.
