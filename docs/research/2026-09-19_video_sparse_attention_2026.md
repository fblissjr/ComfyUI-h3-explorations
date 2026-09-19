# Four training-free video sparse-attention methods of 2026, read for MiniMax H3

> **Provenance, 2026-09-19.** Written by a research subagent of the `libguy`
> session on the lead's brief (item 3 of the second brief), read-only, no GPU.
> The commissioning session re-checked its repo pointers and found them
> correct: the exact stage following routed density (`docs/SOLATTN.md:1029-1031`),
> Sol's densities on the covered-market cells
> (`bench/results/2026-09-18_sol_block_size.md:40-45`) and that sweep's
> density convention (`bench/sweep_sol_block_size_on_capture.py:104-121,321`),
> and the `tau_profile` and pooled-tail lines in `docs/SOLATTN.md`. Paper
> figures are the papers' own, on their setups. The LoSA capture test in
> section 1(h) needs the GPU and is a proposal, not a run.

Provenance: research subagent, 2026-09-19, read-only, no GPU, no git. All four
papers were read in full from arXiv HTML (LoSA also as PDF, to check for an
appendix: there is none). SVOO's code was read at a pinned commit, downloaded
as a zip, not cloned. HASTE v1 was read as an abstract only (arXiv has no HTML
for v1); its v2, retitled HEART, was read in full. No public code exists for
LoSA, HEART/HASTE or DynSparse (searched below). The local capture sets were
listed by name and their workflow and manifest JSON read. No tensor was
opened. Tags: [V] read in paper text or code, with a pointer. [I] inference.
[?] not verified, with what was searched. Arithmetic from shapes is labelled
as arithmetic.

## What this means for H3, ranked

1. **LoSA first, but as a question about the selection rule, not about speed.**
   [I] Its premise is that one exact block-mass pattern, built at an early
   dense step, keeps its retained mass for the rest of the trajectory. That
   premise can be tested offline on the captures we already have. Freezing the
   mask saves almost nothing on our kernel, because Sol's routing and pooled
   tail are already negligible in cost and the exact stage follows routed
   density (`docs/SOLATTN.md:1029-1031`). LoSA's default operating point is
   also much denser than Sol's: 66.1% block coverage at theta 0.99 (LoSA
   Table 3, Wan2.1-1.3B), against Sol's routed density of 0.202 to 0.227 on
   the covered-market step-15 cells at tau 1.0
   (`bench/results/2026-09-18_sol_block_size.md:40-45`). That sweep's density
   counts everything attended exactly, forced diagonal and sinks included
   (`bench/sweep_sol_block_size_on_capture.py:104-121,321`), so it is the same
   convention as LoSA's coverage. The gap is about 3x, by arithmetic across
   two different models and block geometries. What might
   transfer is the rule itself. At equal density, does a set chosen from exact
   masses at the last dense step keep more mass at steps 8 to 15 than Sol's
   per-step pooled proxy? LoSA's own Table 2 shows that a mass target beats a
   fixed budget at equal coverage. It also supplies the per-head budget rule
   that the prior survey says nothing in the pipeline produces
   (`docs/research/2026-09-17_rotation_and_lowbit_attention_survey.md:1127-1135`).
   The concrete test is in section 1(h).
2. **HEART's drift signal is the cheapest replication, and it comes out of the
   same pass.** [I] It compares pooled-q/k L1 drift between captured steps
   with the overlap of the masks. The paper reports Spearman -0.7568 for the
   pooled proxy (HEART section 4.2). The per-head threshold calibration (EBC)
   costs far too much for this repo: 33,600 isolated-head full forwards at our
   shape (arithmetic, section 3(g)).
3. **SVOO: test the input-stability claim, and do not port the
   co-clustering.** [I] A per-(block, head) density profile compared across
   two captured scenes would justify a static per-block `tau_profile`, a
   socket Sol already has (`docs/SOLATTN.md:435`). The co-clustering needs
   per-head permutations and variable-size blocks, which Sol's fixed 64-token
   kernel cannot express. The paper and its own code also disagree on the
   selection rule (section 2(b)).
4. **DynSparse last.** [I] Its in-kernel skip needs each tile's exact maximum
   score, so QK^T is still computed for every tile. Inside Sol it could only
   skip the exponentials and PV of tiles Sol already routed. Its lambda rule
   also did not carry over between the paper's own two models without
   retuning (arithmetic, section 4(g)).

**Cross-cutting [V/I].** None of the four measures a clip as long as ours. The
paper configurations run from 27,280 tokens (DynSparse, Wan2.2-5B, stated in
its section 2.3) to about 118,800 (SVOO, HunyuanVideo 720p at 129 frames, by
arithmetic from the stated resolution and frame count); H3 is 104,361
(`bench/results/2026-09-03_capture_inventory_base16.json`). None handles a
packed [text | audio | video] self-attention. All four drop unselected
attention outright. Only Sol keeps a pooled tail term. So every "retained
mass" figure in these papers is mass that is lost, while Sol partly
approximates it (`docs/SOLATTN.md:432`).

**One instrument serves three tests [I].** Exact 64x64 block masses from the
captured q/k feed the LoSA test, the HEART drift correlation and the SVOO
cross-scene profile. Build that instrument once.

## ID check

| relayed | arXiv resolves to | verdict |
|---|---|---|
| LoSA, 2608.12032 | "LoSA: Near-Lossless Sparse Attention for Training-Free Video Diffusion Acceleration", Liu, Wang, Wang, Sun, Xu; v1 only, 2026-08-12 | **match** [V] |
| "Attention Sparsity is Input-Stable", 2603.18636 | same title on v2 (2026-05-08); **v1 (2026-03-19) was titled "Training-Free Sparse Attention for Fast Video Generation via Offline Layer-Wise Sparsity Profiling and Online Bidirectional Co-Clustering"**; ICML 2026; method name SVOO | **match** [V] |
| HASTE, 2605.14513 | v1 (2026-05-14): "HASTE: Training-Free Video Diffusion Acceleration via Head-Wise Adaptive Sparse Attention"; **v2 (2026-08-09) retitled "HEART: Exploiting Head Heterogeneity in Sparse Attention for Video Diffusion"**, same six authors | **match, renamed**. v2 read in full, v1 abstract only [V] |
| cumulative energy, 2606.16317 | "Training-free sparse attention based on cumulative energy filtering", Li et al., 2026-06-15; method name DynSparse | **match** [V] |

**A correction for the prior survey's D.9 [V].** D.9
(`docs/research/2026-09-17_rotation_and_lowbit_attention_survey.md:1159-1161`)
makes two errors:

- It describes "DFSAttn" as "layer-wise sparsity profiling + bidirectional
  co-clustering". It also lists the co-clustering paper as a separate entry.
- The co-clustering paper is 2603.18636, SVOO, under its v1 title. DFSAttn is
  2605.23445 (Hu, Gao, He, Yuan; ICML 2026), a different method: Hilbert-curve
  reordering, hierarchical block scoring, and "sparse mask caching with
  adaptive ratios". Code: https://github.com/jessica-hujie/DFSAttn.

DFSAttn was read as an abstract only. Its mask caching bears on the LoSA
premise, so it is worth a later read. The "HASTE (2605.14513)" listing at
`:1166` and `:1297` names v1 of what is now HEART.

**Adjacent, not read [?].** A web search surfaced arXiv 2608.18484,
"Partition the Support, Reconstruct the Residual: Training-Free Sparse
Attention for Video Generation and World Models". By its title it is the
closest of the 2026 papers seen here to Sol's pooled tail. Only the title and
listing were seen.

---

## 1. LoSA (arXiv 2608.12032)

**(a) Selection unit [V].**

- Block pairs, with query block R = 128 tokens and key/value block C = 32
  tokens in every layer: "the finest granularity that did not compromise
  block-sparse kernel efficiency in our tests" (Experiments, Implementation).
- The sparse kernel is FlashInfer block-sparse. The construction step uses a
  "custom dense-attention kernel that accumulates exact block masses while
  computing full attention".

**(b) What is computed once and what is reused [V].**

- **Mass.** The block mass is the sum of the dense softmax probabilities over
  the query block's rows and the key block's columns:
  `M_t^h(i,j) = sum_{q in B_i^Q} sum_{k in B_j^K} A_t^h(q,k)`, with
  `A = softmax(QK^T/sqrt(d))` row-wise (Motivation, Eq. 1). It is computed per
  layer, per head and per query block. The normalization is per query block:
  each row sums to 1, so a query block's total mass is R, and the target is
  `theta * R` (Methodology, Eq. 3). Nothing is normalized across heads. The
  aggregate recall metric, Eq. 2, pools over heads, but the selection does
  not.
- **Kept set.** For each (layer, head, query block), sort the key blocks by
  mass in decreasing order and keep the shortest prefix whose cumulative mass
  reaches `theta * R` (Eq. 3). This is greedy by mass until the target. It is
  stored "separately for the conditional and unconditional branches under
  classifier-free guidance". The default is theta = 0.99.
- **Step.** Dense attention runs at steps 0, 1 and 2. The support is built
  "during the dense step t0 = 3", and frozen-support sparse attention is used
  afterwards (Methodology, first paragraph). The stated reason is exact
  masses "rather than the coarse importance estimates that per-step sparse
  methods must rely on", built "after the rapidly changing early denoising
  steps" (Introduction).
- **Indexing.** Fig. 2(c) labels the reference mask "Step 4" while its
  caption says "built at t0 = 3". That is an indexing difference: the
  construction step is the 4th evaluation. [V from the figure image]
- **Schedule length.** The Fig. 2(b) axis runs to 50 denoising steps. So the
  paper runs 4 dense evaluations of 50 [V figure; 8% by arithmetic].
- **Un-kept blocks** are dropped from the numerator and the denominator: the
  softmax is renormalized over the retained keys (Eq. 4). "We do not impose
  auxiliary local or diagonal blocks."
- **Refresh.** The mask is never refreshed. The frozen block indices are
  reused "for all remaining steps", and Q/K/V and the weights are recomputed
  every step (Introduction; Methodology).
- **No t0 ablation.** The ablations cover only the selection rule (Table 2)
  and theta (Table 3). Which step to measure at is never varied. [V]
- **Text and conditioning.** LoSA "is applied to self-attention only, since
  cross-attention accounts for a much smaller fraction of inference cost"
  (Implementation). [?] HunyuanVideo carries its text tokens inside joint
  attention, and the paper does not say whether its mask covered them.
  Searched: the full HTML and PDF text for text/token/conditioning handling.

**(c) Per-head policy [V].** It is implicit in the per-(head, query block)
mass target. Table 2 compares it with fixed-ratio top-k at equal coverage
(66.1%, Wan2.1-1.3B, 100 prompts). Recall averages 96.5% against 99.0%, and
the worst layer 86.6% against 97.3%. In the paper's words, "the uniform
budget truncates exactly those heads whose attention is spread broadly".

**(d) Measured quality cost [V, Table 1; VBench full prompt suite, one fixed
seed per configuration, H200].**

| setting | method | speedup | VBench Overall change vs dense |
|---|---|---|---|
| Wan2.1-T2V-1.3B, 480p | LoSA alone | 1.36x | -0.06 |
| Wan2.1-T2V-1.3B, 480p | SVG2 alone | 1.69x | -0.45 |
| Wan2.1-T2V-1.3B, 480p | D2Cache alone | 2.33x | -0.15 |
| Wan2.1-T2V-1.3B, 480p | LoSA + D2Cache | 2.50x | -0.15 |
| Wan2.1-T2V-1.3B, 480p | LoSA + D2Cache | 3.09x | -0.30 |
| Wan2.1-T2V-14B, 720p | LoSA + D2Cache | 2.50x | -0.43 (SVG2 + D2Cache: 2.46x, -0.71) |
| HunyuanVideo-13B, 540p | LoSA + D2Cache | 3.19x | -0.02 (SVG2 + D2Cache: 3.18x, -0.32) |

- Table 3 (Wan2.1-1.3B, 100 prompts) sweeps theta. At 0.999, 0.99 and 0.95,
  coverage is 83.5%, 66.1% and 46.6%; recall is 99.9%, 99.0% and 95.9%; and
  the speedup is 1.26x, 1.37x and 1.52x.
- Frame counts are not stated ("default sampling configurations"). [?]
- **No long-clip experiment exists.** [V]

**Mask-drift evidence [V].**

- Fig. 2(b) is on Wan2.1-14B, ten prompts, all layers and heads. A 99% pattern
  built at t0 = 3 and frozen keeps a mean recall "close to 99% throughout the
  remaining trajectory", 98.7% at the final step. "Even the worst-performing
  layers retain more than 97%" (Motivation, "Near-Lossless Patterns Are
  Stable").
- Fig. 2(c) is one head. The covered fraction of each later step's top-99%
  region is 99.4% to 99.7% [V, read from the figure image, not the text].
- The paper's explanation is that "the support contracts within an early
  envelope rather than migrating". [I] That reading rests on the one-head
  visualization. The aggregate evidence is Fig. 2(b) only.
- [?] Fig. 2(b) evaluates recall "using the true dense attention mass at
  subsequent steps". Whether the trajectory that produced those later q/k ran
  dense or ran under LoSA is not stated.

**(e) Code.** None found [?]. The arXiv abstract page carries no link. The
full text mentions code only to say AdaSpa's is unavailable. The GitHub
repository search API returned 0 hits for "LoSA sparse attention" and "LoSA
video diffusion" (2026-09-19). A web search found only paper mirrors. Local
clones: the requested grep over `coderef/sglang coderef/vllm-omni
coderef/LightX2V coderef/Sana coderef/diffusers` found no LoSA or
retained-mass code. Its hits outside `.git` were Sana probe policies (sections
2(e) and 3(e)) and one false positive, "hasten" in a text fixture.

**(f) Measured versus claimed.**

- Measured [V]:
  - Frozen-pattern recall across steps: Wan2.1-14B, ten prompts, one t0.
  - Recall, coverage and time for two selection rules and three thetas:
    Wan2.1-1.3B, 100 prompts.
  - VBench at matched speed on three models, one seed each, no variance
    reported.
- Claimed beyond that [I]:
  - That support stability is a property of "video diffusion attention" in
    general. It was measured on one model's 50-step trajectory from one
    construction step.
  - That the method is "near-lossless by construction". The paper itself says
    the guarantee "is exact at t0" only.
  - That LoSA gives the "best training-free speed-quality trade-off across all
    evaluated models". This rests on VBench Overall differences of 0.02 to 0.45
    points from single-seed runs.

**(g) What it would take on frozen H3 [I].**

- **Geometry.** Sol uses 64x64 blocks
  (`coderef/comfy-kitchen/comfy_kitchen/__init__.py:164`, "Each 64-token
  query block"). A LoSA query block spans two Sol query blocks, and a LoSA key
  block is half a Sol key block. Coarser key blocks make the mass prefix less
  selective. Finer query blocks push the other way. So H3's coverage at theta
  0.99 is not predictable from Table 3 and has to be measured.
- **Construction step.** At 16 steps, `start_percent` 0.2 makes 4 of 16
  evaluations dense (`workflows/h3_config.py:553-554`). The step table in
  `docs/SOLATTN.md:1472-1477` reads 11/16 sparse because its header assumes
  `end_percent` 0.9. `end_percent` is 1.0 now, so the dense steps are 0 to 3
  and there is no dense tail. The natural t0 is step
  3, the last dense step. That is the same count of dense steps as LoSA but
  25% of our schedule against their 8% (arithmetic). A later t0 should favour
  stability if the contraction story holds. The dense steps already cost more
  attention time than all the Sol steps together (`docs/SOLATTN.md:1033-1034`).
  LoSA does not reduce them, and it adds a mass-accumulating pass that costs
  "roughly one denoising step" in the paper.
- **Kernel.**
  - Kitchen's `sol_attn` takes a scalar `tau`, `topk_ratio`, `sink_blocks`,
    `key_bias`, `coarse_gate`, `token_aug` and `blk_cnt`. It has no input for
    an externally supplied block map
    (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/__init__.py:2620-2637`;
    `coarse_gate` is a per-token gate for VSA's coarse branch,
    `coderef/comfy-kitchen/comfy_kitchen/__init__.py:191`). A frozen or union
    mask needs a new kernel argument.
  - It also needs a dense pass at step 3 that emits masses. Our dense steps
    run kitchen INT8 attention or sage (`docs/h3_quant_policy.md:27`), and
    neither is documented to emit block masses [?]. Searched only the
    `sol_attn` signature and docstring, not the dense kernels.
- **Storage.** A bitmask over 50 blocks x 56 heads x 1631^2 block pairs is
  about 0.93 GB (arithmetic: ceil(104,361/64) = 1631). LoSA's claim that
  indices are "orders of magnitude smaller than a token-level mask" is true,
  but the mask is not small on a 24 GB card.
- **Tail.** With Sol's pooled tail on, the blocks LoSA drops would be
  approximated rather than lost. That changes what theta buys, so theta would
  need re-deriving under the tail.
- **Conditioning.** Sol forces the conditioning key blocks and the diagonal
  exact (`docs/SOLATTN.md:429`; `bench/analyze_routing.py:22-23`). LoSA forces
  nothing and claims the mass criterion keeps "whichever blocks dominate". H3
  packs text and audio into the same self-attention, so that claim is
  directly checkable.
- **CFG.** H3 runs cfg 1.0 (capture manifest, `sampling.cfg`), so there is
  one mask per layer and head, not two.

**(h) The offline capture test [I, a proposal].**

*Capture sets, as listed on 2026-09-19.* A peer session was writing to the
covered-market directories while they were listed, so list them again before
running.

- **Primary: `2026-09-03_base16_t2v_1344x768`.**
  - The dense-trajectory control: Sol absent, sage throughout (its manifest,
    `workload.attention`).
  - Blocks 0, 24, 32, 40 and 49; steps 4, 8, 12, 14 and 15; S = 104,361;
    seed 730451892; retained under its capture retention record
    (`bench/results/2026-09-03_capture_inventory_base16.json`).
  - It has no step below 4. **Build the mask at step 4**, which ran dense in
    this capture and is one step after the shipped dense window. Evaluate at
    steps 8, 12, 14 and 15.
  - **Run the t0 ablation LoSA never ran here, on this set.** Build Omega
    separately at s4, s8 and s12, and evaluate all three at s14 and s15.
    Equal recall at s15 from t0 = 4 and t0 = 12 supports the premise
    strongly. Recall that degrades with the horizon means the premise is
    horizon-limited, and a periodic refresh (HEART's or SVOO's) is indicated
    instead of a single freeze. It uses the same masses, subset differently.
  - This set tests the premise on a clean trajectory
    (`docs/research/2026-09-03_sol_exact_pquant_and_base_capture.md:197-219`
    for why a dense-trajectory control differs from production).
- **Production-trajectory confirmation:
  `2026-09-19_covered_market_sage_chain_steps` plus `_depth`.**
  - Same seed and S as base16 (`workflow_api.json`: RandomNoise 730451892;
    S from the file names).
  - Sage fp8++ on the dense steps and Sol at tau 1.0 after them.
  - `_steps` holds s1 and s3 as plain files (dense steps) and s4 and s12 as
    `_ksol` files. `_depth` holds s8 and s15 as `_ksol` files.
  - Blocks in common across s1 to s15: 0, 24, 32, 40, 45, 48 and 49.
  - **Build at s3, the real construction step, and at s1. Evaluate at s4,
    s8, s12 and s15.** The plain/`_ksol` boundary is the dense/sparse
    boundary, exactly where LoSA builds, so building on plain files and
    evaluating on `_ksol` files is the production design, not a mix-up.
  - Caveat: neither directory has a `manifest.json` yet, and `_steps` and
    `_depth` may be separate renders. Confirm they share one render, or one
    deterministic graph, before comparing across them. Later steps here
    follow a Sol trajectory, which is the population we actually render.
- **Second scene: `2026-09-10_sol_impl_courtroom`.** A dense trajectory (Sol
  absent per its manifest); blocks 0, 24 and 49; steps 5, 6, 12 and 13;
  S = 104,486. Build at s5 and evaluate at s6, s12 and s13.

*Computation, per captured (block, step), per head.*

- **Use all 56 heads.** The per-head budget question is a tail question: D.7's
  figure is the 5th percentile of mass recall. Per-head behaviour spreads
  widely within one block and step (`bench/analyze_sol_error.py:28-30`). The
  first 8 heads are not a sample of that distribution. A head prefix of 8, as
  in the block-size sweep (`bench/results/2026-09-18_sol_block_size.md:8`),
  answers only frozen against oracle, not the per-head budget question.
- **Masses are not the constraint; reading the cells is.** 596 MB fp32 of
  masses for 56 heads, by arithmetic, against about 4.18 GiB per q/k/v cell
  (`docs/research/2026-09-03_sol_exact_pquant_and_base_capture.md:200-201`).
  Keep each cell resident and run both passes on it before releasing it: a
  naive two-pass reads the capture set twice.

1. **Exact 64x64 block masses in fp32.** Use two tiled passes over the keys.
   Pass 1 is the row log-sum-exp over all 104,361 keys. Pass 2 accumulates
   `exp(s - lse)` into block sums. Never materialize the attention matrix.
   - Shapes (arithmetic): 1631 blocks, the last holding 41 tokens.
     2,660,161 block masses per head: 85 MB fp32 for 8 heads, 596 MB for 56.
     About 2.79e12 FLOP of QK^T per head per pass.
   - This is a GPU job when the owner allows one.
2. **Build the frozen set** Omega^{h,i}_{t0}. For each query block, sort by
   `M_{t0}(i, .)` and keep the shortest prefix reaching `theta * rows(i)`, for
   theta in {0.95, 0.99, 0.999}, LoSA's Table 3 grid. Here rows(i) is 64, or
   41 for the last block.
3. **Primary statistic.** Compute frozen recall
   `R_t = sum_{i, j in Omega} M_t(i,j) / sum_{i,j} M_t(i,j)` at each later t.
   Report it two ways. First, pooled over heads, as LoSA's Eq. 2 does, so it
   compares like for like with Fig. 2(b). Second, per head, with the mean, 5th
   percentile and minimum over query blocks: that view carries the per-head
   budget question.
4. **Comparators at the same per-(head, query block) cardinality |Omega|.**
   - (a) **Sol's own routed set at step t. This arm carries the decision.**
     Take it from the eager reference that `bench/analyze_routing.py`
     already wraps. Choose tau so that its per-head block count matches
     sum |Omega|: compare at equal routed density, never at equal tau
     (`bench/results/2026-09-18_sol_block_size.md:23`).
   - (b) The oracle: the top-|Omega| blocks by `M_t`. The drift cost is
     oracle recall minus `R_t`.
   - (c) A structure-matched floor: a diagonal band plus the conditioning
     blocks, with the same cardinality. A uniform random set, with expected
     recall |Omega|/n_k by arithmetic, is only a sanity check. Attention's
     diagonal structure clears it trivially, and `docs/evidence.md:78` is
     about a control matched on the right axis, not just any control.
5. **Coverage.** Compute `sum |Omega| / (n_q n_k)` at theta 0.99, per head,
   next to Sol's density on the same cells. This decides whether LoSA could
   ever be a speed candidate on H3.
6. **Nesting (Fig. 2(c)'s statistic).** Compute the fraction of step t's own
   top-theta mass that falls inside Omega_{t0}, and `|Omega_t| / |Omega_{t0}|`.
   The contraction story predicts a covered fraction near 1 and a shrinking
   ratio.
7. **Conditioning.** Compute the share of the conditioning key blocks (the
   ranges `sink_conditioning` forces) that Omega includes unforced, and the
   mass on them.

*What falsifies the premise.* Judge against the paper's own curve, not a new
threshold. Fig. 2(b) shows a mean near 99%, 98.7% at the final step, and the
worst layers above 97%.

- **Falsified on H3** if the pooled `R_t` at s12 and s15 falls clearly under
  that curve while the oracle at the same cardinality stays near theta. Then
  the loss is drift, not the rule.
- **Also falsified** if `R_t` falls steadily with step distance, or with the
  t0-to-t horizon in the base16 ablation, toward the structure-matched floor
  rather than toward theta.
- **Also falsified** if the step-t top-theta sets fall materially outside
  Omega_{t0}. Fig. 2(c) shows 99.4% to 99.7% covered for one head.
- **The premise holds but gives Sol nothing** if comparator (a), Sol at
  equal density, matches or beats the frozen set at s8 to s15. That is a
  clean negative for any kernel work.
- **Only if the frozen or union set beats Sol's per-step set at equal density
  does a kernel argument (g) become worth building.** Even then, the next step
  is an end-to-end render. The prior survey records that single-layer error
  misorders estimators relative to 40 denoise steps
  (`docs/research/2026-09-17_rotation_and_lowbit_attention_survey.md:1136-1143`).

---

## 2. SVOO, "Attention Sparsity is Input-Stable" (arXiv 2603.18636, ICML 2026)

**(a) Selection unit [V].**

- Variable-size clusters per head: K_q = 256 query clusters and K_k = 1024 key
  clusters (section 5.1). Tokens are permuted per head so that each cluster is
  contiguous.
- Attention runs block-sparse over (q-cluster, k-cluster) pairs, using
  "dynamic block-size FlashInfer kernels" (Kernel Customization).
- In code, the permutation is applied at
  https://github.com/Mutual-Luo/SVOO/blob/e4ae67b579766bcbe820bda7d34e104ff4c82d5f/svoo/models/wan/attention.py#L1037-L1042
  and inverted by an argsort of the sorted indices at
  https://github.com/Mutual-Luo/SVOO/blob/e4ae67b579766bcbe820bda7d34e104ff4c82d5f/svoo/models/wan/attention.py#L730-L731.

**(b) What is computed once and what is reused [V].**

- **Offline, per (layer, head).** The density is the minimum fraction of
  entries covering tau = 0.95 of the cumulative mass per row, averaged over
  rows (Eq. 5), over m random calibration inputs. The method fits a Gaussian
  and takes the upper alpha = 0.95 quantile, with s = 1 - d-hat (Eq. 6). The
  method text never states m. Fig. 2 used 5 VBench prompts (section 3.1).
- **"Input-stable" as measured** is per-layer density at 80% mass, plotted
  for 5 inputs per model (Figs. 2 and 10) and across steps (Fig. 11). No
  numeric stability statistic is tabulated. Theorem 4.2 bounds the difference
  in logit variance, a proxy, not the density.
- **Online co-clustering (Algorithm 1).**
  - Initialize with random token anchors.
  - Step A: represent each key by its affinity profile to the query centroids
    (`K C_q^T`, L2-normalized), assign it to the nearest key-centroid profile,
    and update the key centroids.
  - Step B: do the same for the queries against the key centroids.
  - It runs for 2 iterations and is "recomputed every N steps", with N = 20
    in section 5.1.
  - Code:
    https://github.com/Mutual-Luo/SVOO/blob/e4ae67b579766bcbe820bda7d34e104ff4c82d5f/svoo/co_clustering.py#L2402-L2451
- **Selection in the paper** is Eq. 7: `rho = min(Recall(A-bar, tau), s)` if
  s > theta, else max(Recall(A-bar, tau), s), with theta = 0.1. The estimate is
  `A-bar = C_q C_k^T`.
- **Paper and code diverge. [V code; I that it is a divergence]**
  - The shipped selection is top-p 0.90 over a cluster-size-weighted softmax
    of the centroid scores, plus a per-head floor on the kept fraction of key
    clusters
    (https://github.com/Mutual-Luo/SVOO/blob/e4ae67b579766bcbe820bda7d34e104ff4c82d5f/svoo/co_clustering.py#L556-L599).
  - The floor is read from the offline profile per (step, layer, head) and
    clipped to [0.15, 0.20] for Wan2.1
    (https://github.com/Mutual-Luo/SVOO/blob/e4ae67b579766bcbe820bda7d34e104ff4c82d5f/svoo/models/wan/attention.py#L931-L956;
    https://github.com/Mutual-Luo/SVOO/blob/e4ae67b579766bcbe820bda7d34e104ff4c82d5f/scripts/inference/wan/wan_t2v_720p_svoo.sh#L72-L84).
  - The profile keeps the maximum over prompts, not a Gaussian quantile
    (https://github.com/Mutual-Luo/SVOO/blob/e4ae67b579766bcbe820bda7d34e104ff4c82d5f/svoo/sparsity/merge.py#L74;
    https://github.com/Mutual-Luo/SVOO/blob/e4ae67b579766bcbe820bda7d34e104ff4c82d5f/scripts/offline/README.md#L74-L76). The shipped profiling prompt file
    has one line (https://github.com/Mutual-Luo/SVOO/blob/e4ae67b579766bcbe820bda7d34e104ff4c82d5f/data/profile_data/prompt.txt).
- **Reuse, in code.** It reclusters at step 11, then every 20 steps (1.3B) or
  every 40 steps (14B) of 50 (https://github.com/Mutual-Luo/SVOO/blob/e4ae67b579766bcbe820bda7d34e104ff4c82d5f/scripts/inference/wan/wan_t2v_720p_svoo.sh#L81-L82,
  https://github.com/Mutual-Luo/SVOO/blob/e4ae67b579766bcbe820bda7d34e104ff4c82d5f/scripts/inference/wan/wan_t2v_720p_svoo.sh#L102-L103;
  https://github.com/Mutual-Luo/SVOO/blob/e4ae67b579766bcbe820bda7d34e104ff4c82d5f/svoo/models/wan/attention.py#L688-L705). With 10 dense steps, Wan2.1-14B clusters at steps
  10 and 11 and reuses that result for steps 12 to 49 (arithmetic from the
  config). During reuse the map is recomputed from the cached centroids, so it
  changes only through the per-step floor. [I from
  https://github.com/Mutual-Luo/SVOO/blob/e4ae67b579766bcbe820bda7d34e104ff4c82d5f/svoo/models/wan/attention.py#L705-L718 and
  https://github.com/Mutual-Luo/SVOO/blob/e4ae67b579766bcbe820bda7d34e104ff4c82d5f/svoo/models/wan/attention.py#L1079-L1086] In effect this is a LoSA-like freeze from an early
  sparse step, chosen by design, not measured here. Fig. 8 reports high
  mutual-information similarity of the clusterings across steps; the values
  are not tabulated.

**(c) Per-head policy [V].** Per head throughout: the profile, the floor and
the clustering.

**(d) Measured quality cost [V, Table 1 (T2V), H200; 81 frames for Wan and 129
for HunyuanVideo, 720p; dense warm-up of 20% of steps for Wan and 10% for
HunyuanVideo; first layer dense (section 5.1)].**

| model, 720p T2V | SVOO speedup | PSNR / SSIM / LPIPS vs dense | SVG2 for comparison |
|---|---|---|---|
| Wan2.1-1.3B | 1.93x | 29.986 / 0.898 / 0.125 | 1.73x, 29.268 / 0.886 / 0.127 |
| Wan2.1-14B | 1.64x | 27.786 / 0.893 / 0.111 | 1.57x, 27.342 / 0.892 / 0.111 |
| Wan2.2-14B | 1.63x | 24.846 / 0.860 / 0.144 | 1.52x, 24.477 / 0.856 / 0.142 |
| HunyuanVideo-13B | 2.17x | 24.879 / 0.843 / 0.224 | 1.96x, 25.218 / 0.841 / 0.205 |

- On HunyuanVideo, SVOO wins on speed, and SVG2 has the better PSNR and LPIPS.
- Attention recall, SVOO against SVG2's k-means (Fig. 9, read from the PDF's
  figure text, SVOO first): 84.29/79.78, 90.98/88.51 and 92.50/89.90 on
  Wan2.1-1.3B, Wan2.1-14B and Wan2.2-14B.
- [I] The HunyuanVideo 720p, 129-frame run is about 118,800 tokens, by
  arithmetic assuming a 4x8x8 VAE and 1x2x2 patching. That is the only
  configuration among the four papers at or above H3's 104k, but at an
  assumed 24 fps it is about 5.4 s of video against our 14.4 s (345 frames at
  the capture manifest's 24 fps).

**(e) Code.**

- Public [V]: https://github.com/Mutual-Luo/SVOO, head
  `e4ae67b579766bcbe820bda7d34e104ff4c82d5f` (2026-06-09). The repository also
  carries an "SVOO-EAR" integration adapted from SVG-EAR, which is not in the
  paper (https://github.com/Mutual-Luo/SVOO/blob/e4ae67b579766bcbe820bda7d34e104ff4c82d5f/README_EAR.md).
- Local [V]: Sana's `qk_coclustering` is a Cosmos3 probe built on SpargeAttn
  mean-similarity block maps, not SVOO's algorithm
  (`coderef/Sana/config/sparse_attention/qk_coclustering.toml:3-14`;
  `coderef/Sana/techniques/sparse_attention_policies.py:1196-1206`; the idea
  is sketched at `coderef/Sana/search_space/04_sparse_attention.md:97-107`).

**(f) Measured versus claimed.**

- Measured [V]: density-per-layer curves on 5 inputs; the recall comparison
  in Fig. 9; clustering similarity in Fig. 8; the end-to-end tables.
- Claimed [V/I]:
  - "Sparsity is an intrinsic layer property." It is supported visually on 5
    inputs, with no statistic.
  - "The calibration set can be arbitrary." Not tested.
  - "Further analyses of reuse steps, block numbers, and offline-stage
    computational overhead are provided in Appendix B." The v2 Appendix B
    contains only B.1, in both the HTML and the PDF.
  - The shipped code does not implement Eq. 7 as written.

**(g) What it would take on H3 [I].**

- **Co-clustering** needs a per-head permutation of q, k and v and a
  variable-size block kernel. Sol's kernel uses fixed, contiguous 64-token
  blocks. The prior survey lists SVG2 and FlashInfer's block-sparse kernels as
  not running on this card
  (`docs/research/2026-09-17_rotation_and_lowbit_attention_survey.md:974-978`;
  not re-verified here for SVOO's variable-block path [?]). The nearest thing
  on our side is `morton`, one global reorder that is not per head and not
  coupled between q and k (`docs/SOLATTN.md:430`).
- **Clustering cost** is per head and per iteration pass, on the order of an
  N x 1280 x 128 GEMM, against attention's N x N x 128. That is about 1.2% of
  QK^T at our N (arithmetic).
- **The profiling half** maps onto Sol's per-block `tau_profile`
  (`docs/SOLATTN.md:435`). A per-head profile needs a kernel change, because
  `tau` is a scalar
  (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/__init__.py:2624`).
- **Offline premise test** (from the block masses of section 1(h)):
  - Compute per-(block, head) density-at-mass at 64x64, at 0.95 and at 0.80,
    on base16 (covered market) against the courtroom set at step 12, for
    blocks 0, 24 and 49. Both are dense trajectories, and S differs slightly:
    104,361 against 104,486.
  - Report the per-head rank correlation within each block and the maximum
    absolute difference, next to the spread across blocks.
  - **Falsified** if the differences between scenes are comparable to the
    differences between blocks: then no static profile is justified. Two
    scenes are thin; the paper used five.

---

## 3. HEART, formerly HASTE (arXiv 2605.14513; v1 HASTE 2026-05-14, v2 HEART 2026-08-09)

**(a) Selection unit [V].** The unit is inherited from the backbone
(section 8.1):

- XAttention: 128-token blocks with antidiagonal scoring at stride 16, and
  top-p at 0.9.
- SVG2: 300 query and 1000 key centroids, p = 0.9, and a minimum kept
  key-cluster ratio of 0.10.

HEART changes only which threshold each head gets and when each head's mask is
recomputed.

**(b) What is computed once and what is reused [V].**

- **EBC (offline, once per model-backbone pair).**
  - Each head is probed with thresholds {0.85, 0.90, 0.95}, one step sampled
    from each of [0,11], [12,24], [25,36] and [37,49] of 50, on one prompt from
    a shared set of 10 VBench prompts. All other heads stay dense.
  - The error is the band-weighted 3D-FFT energy of the velocity error, with
    weights 1.0, 0.5, 0.01 and 0.01 for LL, LH, HL and HH.
  - An ILP (PuLP/CBC) picks one threshold per head under a global sparsity
    budget matched to the shared p = 0.9. That is 12 probes per head, 12LH in
    total (sections 4.3, 7.4, 8.2).
- **TMR (online, per head).**
  - Each head keeps an anchor step's mask until the drift
    `||Q-bar_ta - Q-bar_tb||_1 + ||K-bar_ta - K-bar_tb||_1` exceeds delta.
    The bars are token means, one vector per head.
  - delta is 30 for XAttention on Wan, 8 for SVG2, and 30 for SVG2 on
    HunyuanVideo.
  - Layer gating: a layer is fully reused below a 0.4 refresh ratio and fully
    refreshed above 0.8 (sections 4.2, 8.2).
  - The pooled drift correlates with adjacent-step mask IoU at Spearman
    -0.7568. Full-token drift gives -0.7914 (section 4.2).

**(c) Per-head policy [V].** Both halves work per head: the threshold is
chosen offline, and the refresh decision is made online.

**(d) Measured quality cost [V, Table 2; 81 frames on Wan and 129 on
HunyuanVideo at 720x480; 50 steps; first five steps dense; A800].**

- Wan2.1-1.3B, 720p:
  - XAttention plus HEART reaches 1.85x end-to-end and 2.68x attention-only,
    against XAttention's 1.67x and 2.23x.
  - PSNR is 22.69 against 22.24. VBench is 75.19% against 75.08%, with dense
    at 75.52%.
- HunyuanVideo-13B: XAttention plus HEART has PSNR 28.92 against 26.06, at
  1.32x against 1.27x end-to-end.
- Ablations (Table 4, Wan2.1-1.3B 480p, XAttention):
  - delta 30 gives PSNR 21.61 at 1.44x. delta 100 gives 20.49 at 1.34x:
    stale masks cost both fidelity and speed.
  - **"Warmup reuse", PAROAttention's freeze-after-warm-up, gives 21.38 at
    1.44x, against TMR's 21.61 at the same speedup.** Interval reuse gives
    21.19 at 1.50x.
- [I] That warmup-reuse row is independent evidence bearing on LoSA's
  premise. Freezing after warm-up lost a little similarity against adaptive
  refresh at matched speed on this backbone; it did not collapse. [?] HEART
  does not say at which step warmup reuse freezes.
- No long clip was run: the longest configuration is about 75,600 tokens
  (Wan 720p, 81 frames, by arithmetic).

**(e) Code.** None found [?]. Neither arXiv version links code. The GitHub
repository search API returned 0 hits for "HEART sparse attention video" and
"HASTE sparse attention", and a web search found none. Local [V]: Sana ships
Cosmos3 probe policies named `headwise_adaptive_budgets` and
`online_mask_search_reuse`, a SpargeAttn map reused under a drift gate. These
borrow HASTE's framing, not its algorithm
(`coderef/Sana/techniques/sparse_attention_policies.py:1157-1175`, `:1303-1316`;
`coderef/Sana/search_space/04_sparse_attention.md:109-116`).

**(f) Measured versus claimed.**

- Measured [V]: IoU heterogeneity per head, for heads from a randomly selected
  layer (Fig. 2), and per prompt and per layer (Fig. 5); the drift-to-IoU
  correlations; per-head threshold responses on XAttention with Wan2.1-1.3B
  at 480p (Fig. 3); the end-to-end tables.
- Assumed or claimed [V/I]:
  - The error is treated as additive across heads, "a measurement-driven
    additive surrogate" (section 9.2). It was never checked against joint
    sparsification, except through the end results.
  - "No sparse-kernel changes" holds for its two backbones.

**(g) What it would take on H3 [I].**

- **TMR** saves mask-prediction time, which Sol barely spends
  (`docs/SOLATTN.md:1029-1031`). Its drift signal is still useful as a
  per-head refresh rule for a frozen-set hybrid. delta is in absolute L1
  units of pooled q and k, so it would need re-deriving for H3.
- **EBC at our shape** is 12 x 50 x 56 = 33,600 isolated-head full forwards
  at S of about 104k. A 16-step render at cfg 1.0 is 16 forwards, so that is
  about 2,100 renders' worth (arithmetic). Sol also has no per-head threshold
  (`tau` is scalar; `tau_profile` is per block).
- **Offline test** (from the same block masses as section 1(h)):
  - Compute the per-head pooled drift for the captured step pairs s3 to s4,
    s4 to s8, s8 to s12 and s12 to s15. Correlate it, per block, with (i) the
    IoU of Sol's routed sets from the eager reference and (ii) LoSA-style
    recall of the earlier step's theta set at the later step.
  - **Replicates** if the rank correlation on H3 is of the paper's sign and
    order. **Falsified** if it is weak. The steps are not adjacent, which
    makes the test harder than the paper's.

---

## 4. DynSparse, cumulative-energy filtering (arXiv 2606.16317)

**(a) Selection unit [V].** The unit is the FlashAttention KV tile: Bc = 128
on Wan2.2-14B and 32 on Wan2.2-5B, "to reach comparable sparsity"
(section 4.2.1). The decision is made inside the FA inner loop (section 3.1).

- The metric is `t_i^j = m-hat_i - LSE_i^{(j-1)}` (Eq. 5): the tile's local
  maximum score against the running log-sum-exp of the tiles kept so far
  (Eq. 6). A tile is skipped when `t < lambda`.
- The recommended setting is `lambda = ln(P_real) + ln(Bc/L)` (Eq. 7).
- [?] The paper does not say how per-row quantities reduce to one decision
  per tile.

**(b) What is computed once and what is reused [V].** Only lambda, a global
constant. Everything else is per tile, per step, online. Nothing is reused
across steps, layers or heads. Its cost is "no additional computation to
estimate the mask".

**(c) Per-head policy [V].** None explicit. Adaptivity comes from each row's
running LSE.

**(d) Measured quality cost [V, Table 1].**

- Wan2.2-14B: 82.14% sparsity at "SpeedUp" 1.61x, against BLASST's 61.42% at
  1.32x.
- VBench against FA3: AQ 0.697 against 0.695, IQ 0.665 against 0.709, SC
  0.879 against 0.911.
- Wan2.2-5B: 81.38% sparsity at 1.61x.
- The sequence lengths are stated: 27,280 (5B) and 75,600 (14B)
  (section 2.3).
- [?] Resolution, frame count, step count and GPU are not stated. Nor is
  whether 1.61x is attention-only or end-to-end.
- Internal inconsistencies [V]:
  - Table 1 gives BLASST's BC as 0.000.
  - The abstract's "approximate 15% in attention computation" is garbled.
  - The ablation text says "when lambda <= -6 the accuracy loss is
    unacceptable", while its own Table 2 shows -8 and -9 as acceptable.

**(e) Code.** None found [?]: no arXiv link, 0 GitHub repository-search hits
for "DynSparse" and "cumulative energy sparse attention", nothing in the web
search, nothing in the local grep.

**(f) Measured versus claimed.**

- Measured [V]: sparsity and VBench on two Wan2.2 models; a lambda sweep on
  5B; MAE against recall by kurtosis bin on 5B (Appendix Table 3).
- Claimed [V/I]: "optimal sparsity", which rests on a variance bound under
  i.i.d. V; and "no additional overhead".

**(g) What it would take on H3 [I].**

- **What it can skip.** Eq. 5 needs each tile's exact maximum, so QK^T runs
  for every visited tile. Only the exponentials and PV of a skipped tile are
  saved.
- **Where it would apply.** Inside Sol that is a second-level skip within the
  routed blocks, about a fifth of the blocks on our cells. The unrouted
  remainder is already pooled.
- **Visit order.** The skip needs a large running LSE, so the diagonal and
  sink tiles would have to be visited first.
- **lambda does not transfer.**
  - The 14B model's lambda = -6 at Bc 128, L 75,600 implies P_real of about
    1.46.
  - The 5B model's lambda = -9 at Bc 32, L 27,280 implies about 0.105.
  - For H3, ln(64/104,361) is about -7.40.
  - (All arithmetic from Eq. 7.) A fresh sweep would be needed.
- **Offline test.** For each query block, replay its routed key blocks in the
  kernel's visit order on captured q and k. Apply `t < lambda` over a grid, and
  measure the skipped fraction and the change in output error against exact
  attention. This needs a separate simulation, not the block-mass pass.
  **A non-starter** if the skipped fraction at Sol's current error is small.

---

## Synthesis: which first, and how each relates to Sol

**Test LoSA first [I].** Its premise decides whether any frozen,
exact-mass-derived routing can beat Sol's per-step proxy at equal density.
The same pass yields the HEART drift correlation and the SVOO cross-scene
profile. DynSparse needs a different simulation and has the least to offer.

| Sol mechanism | LoSA | SVOO | HEART | DynSparse |
|---|---|---|---|---|
| **tau-sigma threshold**, per (head, query block), from pooled centroids, each step (`bench/analyze_routing.py:9`; `docs/SOLATTN.md:425`) | theta-mass prefix per (head, query block) from exact masses at one step, then frozen | top-p on cluster centroids plus a per-head floor from an offline profile (code); Eq. 7 (paper) | per-head top-p threshold chosen offline by velocity error | per-row running-LSE threshold inside the kernel |
| **pooled tail** (`docs/SOLATTN.md:432`) | drops and renormalizes | drops (the EAR code path compensates; not in the paper) | inherits the backbone: drops | skips |
| **token_aug** (`docs/SOLATTN.md:436`) | nothing comparable | per-token cluster membership is the nearest idea | none | none |
| **start_percent dense warm-up**, 4 of 16 (`workflows/h3_config.py:553-554`) | 4 dense of 50, and the last one is used to build the mask: the only one of the four that turns the warm-up into information | 20% (Wan), 10% (HunyuanVideo) | first 5 of 50 | not stated [?] |
| **sink_conditioning, diagonal forced** (`docs/SOLATTN.md:429`) | forces nothing; the mass rule is claimed to be enough | none | none | none |
| **per-block `tau_profile`** (`docs/SOLATTN.md:435`) | per (layer, head, query block) by construction | per (step, layer, head) profile | per head | none |

## Sources

Papers (arXiv; read in full unless noted):

- LoSA: https://arxiv.org/abs/2608.12032, https://arxiv.org/html/2608.12032v1, https://arxiv.org/pdf/2608.12032 (Fig. 2 image: https://arxiv.org/html/2608.12032v1/motivation_combined.png)
- SVOO: https://arxiv.org/abs/2603.18636v1 (title only), https://arxiv.org/html/2603.18636v2, https://arxiv.org/pdf/2603.18636v2
- HASTE / HEART: https://arxiv.org/abs/2605.14513v1 (abstract only), https://arxiv.org/html/2605.14513v2
- DynSparse: https://arxiv.org/abs/2606.16317, https://arxiv.org/html/2606.16317v1
- DFSAttn (abstract only, for the D.9 correction): https://arxiv.org/abs/2605.23445
- Index: https://github.com/Mutual-Luo/Awesome-Sparse-Attention-Video-Diffusion (README)

Code:

- SVOO at `e4ae67b579766bcbe820bda7d34e104ff4c82d5f`: https://github.com/Mutual-Luo/SVOO (files cited above by blob URL and line)
- Searched with no result (GitHub repository search API and web search, 2026-09-19): LoSA, HEART/HASTE, DynSparse

Local (read-only):

- `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/__init__.py:2620-2637`
- `coderef/comfy-kitchen/comfy_kitchen/__init__.py:164,191`
- `coderef/Sana/techniques/sparse_attention_policies.py:1157-1175,1196-1206,1303-1316`
- `coderef/Sana/config/sparse_attention/qk_coclustering.toml:3-14`
- `coderef/Sana/search_space/04_sparse_attention.md:97-116`
- `docs/SOLATTN.md:425-436,1029-1034,1472-1477`
- `workflows/h3_config.py:553-554`
- `bench/analyze_routing.py:9,22-23`
- `bench/sweep_sol_block_size_on_capture.py:104-121,321`
- `bench/analyze_sol_error.py:28-30`
- `docs/evidence.md:78`
- `docs/h3_quant_policy.md:27`
- `bench/results/2026-09-18_sol_block_size.md:8,23,40-45`
- `bench/results/2026-09-03_capture_inventory_base16.json`
- `docs/research/2026-09-03_sol_exact_pquant_and_base_capture.md:197-219`
- `docs/research/2026-09-17_rotation_and_lowbit_attention_survey.md:974-978,1127-1143,1159-1166,1297`
- Capture sets by name, listed and their JSON read, no tensors opened: `2026-09-03_base16_t2v_1344x768`, `2026-09-10_sol_impl_courtroom`, `2026-09-19_covered_market_sage_chain_steps`, `2026-09-19_covered_market_sage_chain_depth`
