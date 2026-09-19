# In-context conditioning, and an open sweep: what 2025-2026 work says about the reference rows, the prefix and the sinks

> **Provenance, 2026-09-19.** Written by a research subagent of the `libguy`
> session for the lead's wider sweep, read-only, no GPU. The commissioning
> session re-checked its code pointers and found them correct: the forced-exact
> prefix range in `sol_attn_h3.py:934`, and the per-latent-frame time spans in
> ComfyUI core `comfy/ldm/minimax/model.py` (`FRAME_PER_TOKEN`,
> `FRAME_RESCALE`, `_video_t_spans`). Paper figures came through a
> summarising fetch tool, as the agent says below; recheck against the PDF
> before reusing one. The block-grouping tool and the 2026-09-19 capture
> inventories named below were other sessions' work, some uncommitted, when
> this was written.

> **Provenance.** Research subagent, 2026-09-19. Read-only: no git, no file
> edited outside this scratchpad, no GPU. Sources: ComfyUI core's MiniMax H3
> model file (`comfy/ldm/minimax/model.py`, two directories above the repo),
> this repo's `sol_attn_h3.py`, `workflows/h3_config.py`, `docs/SOLATTN.md`,
> `docs/morton.md`, `docs/research/technique_transfer.md`, the 2026-09-17
> survey and the 2026-09-19 question review, and the `coderef/comfy-kitchen`
> and `coderef/sage-fork` trees. Papers: every arXiv id below was opened on its
> abstract page or HTML page this session. **Read at section level through the
> HTML** (the fetch tool returns a model-written digest of the page, not the
> raw text, so every quoted number below should be re-read in the PDF before
> it carries weight): FullDiT2, LIVEditor/ISA, ToPi, Vision-Language Binding,
> DIAL, HALO, Avatar V, Stand-In, OminiControl2, the 2025 VDiT attention
> analysis, the 2026 DiT sink causal analysis, VMonarch, PAROAttention, JoVA.
> **Abstract only:** Elastic-Cache (plus its project summary), KVSink,
> MASQuant, MorphoQuant, Taming Outlier Tokens, Massive Activations (DG),
> In-Context Audio Control, LTX-2, Draft-and-Target Sampling. Nothing was
> measured for this document; the only numbers of my own are arithmetic from
> shapes, labelled as such.

Labels: **[V]** read in code or paper text, with pointer. **[I]** my
inference. **[?]** not verified; says what was searched.

One word needs splitting before anything else. In this pack **"sink"** is the
name of a kernel range: `_sink_blocks` returns key blocks
`[0, ceil(video_start/64))`, the whole conditioning prefix, forced exact
(`sol_attn_h3.py:934`). In the literature an **attention sink** is a measured
phenomenon: a key position that draws a disproportionate share of every
query's softmax mass. They are different objects. The useful question falls
out of keeping them apart: does H3 have measured attention sinks, and do they
sit inside the forced-exact range or out in the video span that Sol routes?

---

## What this means for H3, ranked

Ranked by how directly each could change a default or a kernel in this pack
within a week. "Measure first" prefers the existing captures: set A, covered
market, t2va, one sequence length, blocks 0/8/16/24/32/40/44-49 at steps 8
and 15 plus blocks 0/24/32/40/45/48/49 at steps 1/3/4/12
(`bench/results/2026-09-19_capture_inventory_covered_market_depth.json`,
`bench/results/2026-09-19_capture_inventory_covered_market_steps.json`).
**No capture on disk contains reference rows**: capture-plan set D (ref2va) was
never run (`internal/2026-09-18_capture_plan.md`, set D). Every
reference-specific measurement below waits on that one capture render.

1. **One CPU pass on set A: segment-resolved exact mass and a measured-sink
   map, all 56 heads.** [I] For sampled query rows stratified by segment
   (text, target audio, video by latent frame), the exact fp32 softmax row,
   reduced per (block, step, head) to: mass on text / audio / video; incoming
   mass summed per latent frame index **for every frame**, not binned, so
   that a frame-0 sink, a last-frame sink or a periodic one each shows as its
   own shape in the curve; and the top key rows by total incoming mass, each
   tagged forced-exact or routed and with its value norm. It measures where
   mass goes, not how much error a cell tolerates, so it stays out of the
   excluded tolerance-profiling lane. This is question review C.4's "mass
   outside the diagonal and sink ranges" split by segment, and it must run
   on **all 56 heads**: every
   ordering and error number in the pack so far used a head prefix of 8
   (`internal/2026-09-19_question_review.md` C.4), and a pass whose point is
   "are some heads special" answers nothing on 8. It is the t2v stand-in for
   the reference question (text and target audio are the prefix that exists
   on disk), the audio instrument C.5 says is missing, and the input to items
   2-5. Cost by arithmetic from shapes: 56 heads x 2,048 sampled queries x
   104,361 keys x 128 channels is about 1.5e12 multiply-adds per cell [I],
   CPU. Stream one head at a time: q, k and v for all 56 heads at full length
   in fp32 are several GB resident per cell [I, arithmetic from shapes], and
   the block-grouping tool casts every requested head at once in its loader
   (untracked and changing, so no line number). Home: a new reduction over
   the exact rows that
   `bench/analyze_sol_block_grouping.py` measure B already computes (a peer's
   untracked file; theirs to extend).

2. **Measured sinks in the video span: the one default this could move
   without a kernel change.** [V for the literature, I for H3] In Mochi the
   attention sinks sit in the last four layers, mostly on the **first latent
   frame**, and carry small value norms (arXiv 2504.10317 section 5, Figs.
   9b, 11, 12). VMonarch reports a first-frame sink in Wan2.1 and fixes it by
   recomputing frame 0's queries with full attention (arXiv 2601.22275
   section 3.3, Eq. 8; fine-tuned, not training-free). H3's RoPE gives every
   fifth latent frame a one-pixel-frame time span
   (`comfy/ldm/minimax/model.py:30`, `FRAME_PER_TOKEN = (1, 4, 4, 4, 4)`,
   used at `:95-96`), so H3 may have a sink candidate at every `k mod 5 = 0`
   frame, not only frame 0 [I]. These are two different hypotheses and a
   null on one is not a null on the other: Mochi's is a first-frame anchor
   role (spatially scattered tokens, small value norms), the H3 one would
   come from the short RoPE time span of the one-pixel-frame latents. The
   per-frame mass curve in item 1 tests both; mod 5 is what a periodic curve
   would confirm, not how to bin it. **What would change:** in raster order latent
   frame 0 is contiguous with the forced-exact range (target video is the
   last segment, `comfy/ldm/minimax/model.py:440`), so widening the exact-key
   range by one frame (and, for VMonarch's variant, the dense-query range) is
   a node change in `_sink_blocks`, and raster is the shipped order
   (`morton=False` in `workflows/h3_config.py`'s Sol recipes); periodic
   frames are not one range and would
   need the kernel's single `[start, end)` (`coderef/comfy-kitchen/comfy_kitchen/__init__.py:175-177`) to grow, or a permutation. Under the
   `3d` reorder frame 0 is scattered across cubes, so the same finding would
   also bear on the reorder. **Measure first:** item 1's sink map. If no head
   puts a large share of incoming mass on frame 0 or the periodic frames,
   close this.

3. **The prefix is exact only in the routing sense: its INT8 precision is
   unexamined.** [V for code, I for the risk] "Exact" in Sol means "not
   pooled"; the kitchen lineage still quantises Q, K and V to INT8
   (2026-09-17 survey, section D.2 table). In SD3's joint attention, over
   99.9% of dynamically found attention sinks are text-conditioning tokens
   (arXiv 2605.09313 section 3.6), and LLM KV quantisation keeps sinks at
   full precision for this reason (KVSink, arXiv 2508.04257, abstract). All
   three INT8 paths in play centre K on **one vector for the whole packed
   sequence**: Sol's own preprocess on the per-head mean over every key block
   (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_preprocess.cu:101-115`;
   in the producer path the mean is carried from the previous call,
   `:251-267`); kitchen's dense INT8 SDPA on one anchor key picked from
   **nine evenly spaced rows**
   (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/quant_qk_int8.cu:447-453`), which at full t2v length puts one sample at row 0
   (text) and none in the target-audio segment [I, from the sampling rule and
   the segment order]; sage on the mean over all rows
   (`coderef/sage-fork/sageattention/core.py:967`). Video rows dominate every
   one of those centres. Multimodal-LLM PTQ work names exactly this failure,
   "smoothing misalignment" across modalities (MASQuant, arXiv 2603.04800,
   abstract; MorphoQuant, arXiv 2606.04349, abstract, on an omni model with
   audio). **What would change:** a kernel-side option to keep the prefix key
   blocks in bf16 or centre them on their own anchor. A per-segment centre is
   not free the way a global one is: one vector subtracted from every key
   shifts each query's scores by a constant and cancels in the softmax, but a
   different vector per segment does not, so the kernel would have to add
   `q . mu_segment` back per key segment [I, algebra]. **Measure first:** the
   INT8 attention-output error on set A decomposed by key segment (text /
   audio / video), raster order, with the anchor as shipped, with none, and
   with a per-segment anchor. The existing grader calls the kitchen kernel on
   the GPU (`bench/grade_ck_int8_on_capture.py:14`), so either the card
   is free or the quantiser is re-implemented on CPU.

4. **ref2va: Sol already protects the copy; the open items are the
   reference rows' own queries and the price of forcing every reference key
   exact.** [V for code and papers, I for the ranking] Every target query
   sees every reference key unpooled (`sol_attn_h3.py:934`), so the path the
   literature says carries identity, reference to target directly (FLUX.2
   knockout: blocking reference-to-image attention damages human identity,
   blocking reference-to-text does little; arXiv 2605.24624 Table 2), is not
   approximated by routing. Two things are: (a) under the shipped mode the
   reference rows' **queries** are routed, because the dense-query range is
   target audio only (`sol_attn_h3.py:946`), so the K/V those rows present
   to later blocks carry Sol error, and `docs/SOLATTN.md` calls this
   second-order without a measurement; (b) the forced-exact range grows with
   every reference row, which `docs/SOLATTN.md` ("References are pinned
   exact") prices as the dominant cost at video-reference load. Three 2026
   papers measure reference rows as low-salience and prunable in most blocks
   (ISA, arXiv 2605.04569 Fig. 5; ToPi, arXiv 2602.01609 section on
   representative layers; FullDiT2, arXiv 2506.04213 Fig. 2b). **What would
   change:** the `sink_conditioning` choice on ref2va graphs, and a node-side
   prefix permutation (item 4 of Part 1) that would let references be routed
   while text and target audio stay exact. **Measure first:** capture set D,
   then item 1's pass with the reference segments added.

5. **Caching reference K/V across steps: not a week's win.** [V code, V
   papers, I conclusion] On H3 the reference rows' block-0 input is
   step-invariant, so their block-0 K/V are exactly reusable; from block 1 on
   they depend on the noisy targets through unmasked attention
   (`comfy/ldm/minimax/model.py:199`), as `docs/research/technique_transfer.md` fact 2
   says. Exact reuse is therefore one block of fifty. Every system found that
   caches reference K/V across steps first **trained** one-way attention
   (FullDiT2, OminiControl2, Stand-In, Avatar V; Part 1.4). The training-free
   analogue under symmetric attention is drift-gated approximate reuse from
   diffusion LLMs (Elastic-Cache, arXiv 2510.14973). **Measure first:** step
   drift of the text and audio rows' K and V across set A's steps, per block,
   as the t2v stand-in; references need set D.

6. **Reorder and INT8: PAROAttention's quantisation gain transfers only on
   the K side.** [V code, I transfer] PARO chooses an axis permutation
   offline from a combined sparsity and incoherence metric and isolates the
   reorder's effect on quantisation (arXiv 2506.16054, Table 3, Fig. 13). Its
   headline incoherence reduction is on the attention map P under
   block-wise scales. Kitchen's P is unsigned INT8 against the running row
   maximum with a fixed offset
   (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/attn_utils.cuh:36-40`), a per-row scale that block membership does not
   change, so that part does not transfer [I]. K is quantised per block, so
   the reorder does change which keys share a scale [I]. The ordering sweep
   calls the node's real kernel and so folds INT8 into its error without
   isolating it (`bench/sweep_sol_orderings_on_capture.py` docstring).
   **Measure first:** INT8-K-alone error on set A, raster against `3d`. A
   positive result is an argument for the reorder that the latest decision
   says is missing (`docs/wiki/decisions.md:51`, "`morton` stays off by
   default, and the panel did NOT decide it").

7. **Audio-video synchronisation heads: nobody has published a per-head
   analysis for a single-stream joint sequence.** [?] Searched: "joint
   audio-video diffusion transformer cross-modal attention heads analysis
   synchronization", "single-stream joint audio-video diffusion transformer unified sequence cross-modal attention mass layers 2026",
   "audio-visual joint generation DiT attention map analysis". JoVA uses
   joint self-attention with temporal-aligned RoPE and reports no attention
   analysis (arXiv 2512.13677, section 3.2); LTX-2 is dual-stream
   cross-attention (arXiv 2601.03233, abstract); In-Context Audio Control
   constrains its 3D attention to enforce temporal alignment (arXiv
   2512.18772, abstract). H3 puts audio rows on the video time axis and pins
   the two stereo channels to the frame's left and right `w` extremes
   (`comfy/ldm/minimax/model.py:117-123`) [V]. **Measure first:** item 1's
   audio-to-video and video-to-audio mass per head, with a histogram of the
   RoPE time distance between query and key. Sol already runs target-audio
   queries dense and audio keys exact, so this informs the INT8 and dense
   kernel question (C.5), not Sol.

Not ranked: draft-and-verify sampling (one line under Part 2) and test-time
registers (Part 2.6).

**The first measurement:** item 1 on set A, all 56 heads, steps 1 to 15.
It answers item 2 outright, feeds 3, 5 and 7, and is the template set D's
reference version reuses unchanged.

---

## Part 1. In-context conditioning: where the copy lives and how methods protect it

### 1.1 H3's layout and what Sol does to it [V]

- Packed order for ref2va: `[text | reference blocks | audio | video]`
  (`comfy/ldm/minimax/model.py:5-7`). A video reference packs its audio rows
  immediately before its video rows; an image reference is one frame of rows
  at the reference's own latent size (`:395-436`). Target audio then target
  video are always the last two segments (`:440`).
- Reference timesteps are pinned: `ref_img` at `max(t_v, 0.999)` and
  `ref_audio` at `max(t_a, 1.0)` (`:32-33`, `:635-637`). The visual pin holds
  until the video sigma drops below one part in a thousand, so across a
  schedule the reference rows' modulation row is effectively constant.
  Their input rows are also identical every step: the noise augmentation
  uses a fixed seed and "every condition intentionally restarts the same RNG
  stream" (`:537`). Text rows are not pinned: their modulation follows `t_v`
  (`:635`).
- Every block adds its attention and MLP residual to every row, reference
  rows included, and the attention call has no mask (`:199`, `:291-297`).
- Sol under the shipped mode (`workflows/h3_config.py`,
  `SOL_CORE_DEFAULTS["sink_conditioning"]`, `exact_kv_and_rows` when read):
  key blocks `[0, ceil(video_start/64))` exact for every query, which covers
  text, keyframe cond rows, every reference row and target audio; dense
  queries only for target audio (`sol_attn_h3.py:894-946`). So reference
  **keys** are always unpooled, reference **queries** are routed. The mode
  `exact_kv_and_all_rows` makes reference queries dense too; its price is in
  `docs/SOLATTN.md`, "References are pinned exact".
- The kernel takes one exact-key range and one dense-query range, each a
  single `[start, end)` (`coderef/comfy-kitchen/comfy_kitchen/__init__.py:175-177`). "Text and target audio exact, references routed" is two ranges
  and is not expressible as the layout stands.

### 1.2 Which layers and heads carry the copy

| work | model | what was measured | finding | label |
|---|---|---|---|---|
| Vision-Language Binding (arXiv 2605.24624, May 2026) | FLUX.2, single attention stream over text, reference, noise | attention knockout of reference-to-image and reference-to-text edges past a cutoff layer | identity travels **directly** reference to image (blocking it damages human customization most); colour, style and scene travel **through the text tokens**, bound in the text padding tokens by about the 6th-8th double-stream block. No per-head or per-timestep analysis (Table 2, Appendix A) | [V, HTML digest] |
| ToPi (arXiv 2602.01609, Feb 2026) | FLUX.1 Kontext, Qwen-Image-Edit | offline "context sensitivity score" per layer; per-token influence `||v|| x attention received` | reference influence concentrates in a few layers ({13, 18, 31} of 57 in Kontext; {35, 37, 42} of 60 in Qwen-Image-Edit); total attention to the reference is highest at high noise and decays ("self-contained refinement" late); first 10 layers exempted from pruning | [V, HTML digest] |
| DIAL (arXiv 2609.11507, Sep 2026) | Phantom-14B, Kaleido-14B (both Wan2.1-T2V-14B, references concatenated in-sequence) | per-block spatial correlation between subject masks and attention to each reference | one block gives the most spatially concentrated "intrinsic spatial grounding map"; the map is reliable only in the low-noise regime (roughly the last 80% of steps); heads averaged, block index not disclosed | [V, HTML digest] |
| ISA / LIVEditor (arXiv 2605.04569, ICML 2026) | Wan 2.2 based editor | pooled coarse scores, source queries against source keys and against context keys, by block | source-to-source scores exceed source-to-context, and the gap widens with depth (Fig. 5). No per-head analysis | [V, HTML digest] |
| FullDiT2 (arXiv 2506.04213, 2025) | FullDiT (in-context video conditioning) | cumulative attention from noisy latents to reference tokens; per-block importance; step-wise feature similarity | 50% of reference tokens take over 85% of the attention from noisy latents, averaged over blocks (Fig. 2b); different blocks attend to different reference tokens (Fig. 2c); reference features stable across steps relative to noisy tokens (Fig. 2d); block importance unbalanced (Fig. 2e) | [V, HTML digest]. 2025, included because the 2026 caching work builds on it |
| HALO (arXiv 2607.11081, ECCV 2026) | CogVideoX, checked on Wan | per-head attention entropy and motion displacement | low-entropy "structure" heads (entropy threshold about 7, layers 14-27 used) can take injected reference values without identity leakage; injecting all heads leaks identity. Reference here is a motion video, not an identity image | [V, HTML digest] |

What is **not** in the literature, as far as these searches reach [?]: a
head-level map of reference-to-target identity copying in a video DiT, and
any attention analysis of audio-driven talking video with a reference image.
Searched: "2026 subject-to-video diffusion transformer attention analysis which layers heads transfer identity from reference tokens",
"identity attention heads video diffusion transformer reference subject
consistency head-level analysis 2026", "audio-driven talking avatar
diffusion transformer reference image attention analysis identity layers
2026". Avatar V (arXiv 2606.13872) conditions on a full reference video
in-sequence and presents no per-layer or per-head analysis (HTML digest).
Everything above is layer-level at best.

What it suggests for H3 [I]: the copy is a direct target-to-reference
attention path, concentrated in a few blocks, heaviest early in the schedule
while spatially sharpest late. Sol does not approximate that path (1.1). The
parts Sol does approximate are the reference rows' own queries, and the
prefix's INT8 precision.

### 1.3 How sparse-attention and caching methods treat the reference

- **Prune it, not protect it.** The 2026 in-context methods go the other way
  from Sol's forced-exact range. ISA pre-selects the top fraction of context
  key blocks by pooled score, then routes each query block to full attention
  or a block-wise zeroth-order Taylor term by a "sharpness" statistic
  (arXiv 2605.04569 section 3, Eqs. 1 and 4). That Taylor term is the same
  mechanism as Sol's pooled tail: convergent design, nothing new to import.
  Applied training-free to its own full-attention model it reports 1.47x over
  full attention with equal or better benchmark scores (Table 2, their
  setup). ToPi hard-prunes low-influence reference tokens outside the first
  10 layers, recomputing the mask every 10 steps, and reports about 1.21-1.33x
  (Table 1, their setup), training-free. [V, HTML digests]
- **Keep the reference exact.** Sol's conditioning range is the only design
  found that pins reference keys unpooled at every block and step. [V for
  Sol; ? for others: the sglang and LightX2V H3 reference paths were not
  checked for their sink handling.]
- **Read as a lever for H3** [I]: reference pruning attacks the forced-exact
  floor that `tau` cannot reach (`docs/SOLATTN.md`, the sink table), so it
  would stack with Sol rather than double-count, which is the opposite of the
  worry `docs/research/technique_transfer.md` raises for token merging on video rows. It is
  worth anything only on graphs with large references (the video-reference
  rows in `docs/SOLATTN.md`'s reference table); on image references at
  `match` the prefix is small.
- **The expressible experiment today** [I]: default against
  `exact_kv_and_all_rows` on a ref2va graph measures item 4(a). Routing the
  references while keeping text and target audio exact needs either a second
  kernel range or a node-side permutation of the prefix to
  `[text | target audio | references]` inside the attention call. The
  permutation is output-equivalent after the inverse because RoPE is applied
  before attention (`comfy/ldm/minimax/model.py:181-190`), the same argument
  the video reorder relies on (`sol_attn_h3.py:430`, `_perm_for`). Making the
  reference exact in only a few blocks would additionally need a per-block
  sink setting, which the node does not have
  (`internal/2026-09-19_question_review.md` section D lists what it can set).

### 1.4 Is the reference K/V constant across steps, and does anyone cache it?

**In H3, exactly constant at block 0 only.** [V code, I consequence] The
block-0 input for reference rows is the same patch embedding every step (fixed
seed, 1.1) under the same modulation row (pinned timestep, 1.1), so block 0's
reference K and V are step-invariant. The block-0 attention output of those
rows mixes in noisy target rows (`:199`) and is added to their residual, so
from block 1 on their K and V move with the targets. Exact reuse therefore
covers one block of fifty: at most a fiftieth of the per-step projection work
on reference rows [I, arithmetic from the block count]. And
`docs/research/technique_transfer.md` records the GEMM phases as weight-bytes bound on this
box, so fewer rows through a GEMM may not shorten it. Not worth building.

**Everyone who caches reference K/V changed the attention first, and
trained.** [V, HTML digests]

| system | attention change | trained? | what is cached |
|---|---|---|---|
| FullDiT2 (arXiv 2506.04213) | decoupled: reference queries attend only reference K/V | yes, 400k iterations; the paper calls decoupling "crucial for preventing training-inference misalignment" (Table 2 ablation) | reference K/V at the first step, reused; references processed in 5 of 28 layers |
| OminiControl2 (arXiv 2503.08280) | asymmetric mask, condition tokens do not attend to noisy tokens | yes, "incorporated during model training" (section 3.3); naive reuse without it gives "unsatisfactory results" (Fig. 5). Condition features show high but not perfect step-to-step similarity (Fig. 3) | condition K/V computed once |
| Stand-In (arXiv 2508.07901) | restricted self-attention; reference at timestep zero | yes, LoRA, about 1% extra parameters on Wan2.1-14B (section 4.1) | "we can cache K_I and V_I" (section 3.2) |
| Avatar V (arXiv 2606.13872, 2026) | reference tokens only self-attend | yes | full reference context and per-block reference K/V at the first step (section 4.3) |

So the pattern is: make reference rows independent of the targets, train for
it, then cache. It is `docs/research/technique_transfer.md`'s TaoMate note one level down:
TaoMate re-encodes committed chunks clean and trains for causal attention;
these re-encode the reference clean and train for one-way attention.

**The training-free analogue under symmetric attention** comes from
diffusion language models, which have fact 2 exactly (bidirectional, every
token updated every step). Elastic-Cache reports that KV drift between steps
is small for most steps and grows with depth, and that the most-attended token
drifts least, and refreshes by an attention-aware drift test from a chosen
layer onward (arXiv 2510.14973, abstract and project summary; ICLR 2026 per
its repository). [V abstract-level] No training-free reference-KV cache for a
bidirectional video DiT was found [?; searched "training-free reference KV
cache across denoising steps in-context DiT bidirectional attention" and
"2026 sparse attention in-context video editing reference tokens caching training-free acceleration"].

### 1.5 The cheap measurement on our captures [I]

**Now, on set A (t2v; text and target audio stand in for references):**

1. Segment-resolved exact mass, per (block, step, head), all 56 heads, as in
   ranked item 1. For video queries: share on text, target audio, video, and
   within video the incoming mass per latent frame index, every frame. For
   text queries:
   share on audio and video rows, which is how far the text prefix is from
   one-way attention in this frozen model.
2. Step drift of prefix K and V, per block: relative L2 and cosine of the
   text rows' and audio rows' K and V between set A's captured steps (1, 3,
   4, 8, 12, 15; 3 and 4 are adjacent). If drift is small and grows with depth
   as in Elastic-Cache, prefix reuse is at least plausible; if it is large at
   shallow blocks, close the caching question for H3.
3. Block-0 invariance as a control: the text rows' block-0 K should **not**
   be step-invariant (their modulation follows `t_v`); if a future set D shows
   the reference rows' block-0 K bit-identical across steps, the code reading
   in 1.4 is confirmed.

**After set D (one capture render, `internal/2026-09-18_capture_plan.md`
set D):**

4. The same pass with `ref_img` and `ref_audio` as segments: per head and
   block, the share of target-video queries' exact mass on reference rows
   (where the copy is localised; compare ToPi's few-layer concentration), and
   the share of reference queries' mass on target rows (how far H3 is from the
   one-way attention every caching system trained for; near zero in most
   blocks would make reference K/V nearly constant beyond block 0).
5. Sol error on the reference query rows' outputs, per block, against the
   error on video query rows. This prices item 4(a) before any render; the
   render that settles it is default against `exact_kv_and_all_rows`.

---

## Part 2. The open sweep: what the team's areas do not cover

Excluded by the brief and not repeated: DeepSeek V4.1's indexer, LoSA,
"Attention Sparsity is Input-Stable", HASTE, cumulative-energy filtering,
token-selection scorers, layer/head/step tolerance profiling, step caching,
evaluation metrics. ISA and ToPi appear above as reference-salience and
localisation findings, not as scorers.

### 2.1 Measured attention sinks in the video span (ranked 2)

- **What.** In Mochi, sinks appear only in the last four of 48 layers (over
  80% of heads in the last two show sink patterns), concentrate on the first
  latent frame, are spatially scattered, and have small value norms;
  skipping all sink heads preserves output quality while skipping as many
  random heads degrades it; Hunyuan shows sinks occasionally, Wan and
  CogVideoX not at all (arXiv 2504.10317, sections 5.1-5.2, Figs. 9b-12;
  2025, the base the 2026 work below builds on). VMonarch finds a first-frame
  sink in Wan2.1 that breaks its structured approximation and recomputes
  frame 0's queries with full attention (arXiv 2601.22275, section 3.3,
  Eq. 8; needs fine-tuning). **The two disagree about Wan.** The likely
  reason is definition: 2504.10317 counts single vertical lines in the map,
  VMonarch a frame drawing disproportionate attention [I; neither digest
  compares them]. So whether a video DiT has sinks is a per-model question,
  which is why H3's must be measured, not inherited.
  In SD3's joint attention the sinks are text tokens, peak mid-network and
  early in denoising, and removing them barely moves CLIP-T (arXiv
  2605.09313, sections 3.2 and 3.6, Tables 1-2). [V, HTML digests]
- **Why it is new here.** The pack's forced-exact range covers the prefix by
  construction; nothing checks whether a measured sink lies outside it.
  [V: `sol_attn_h3.py:934`; the question review's C.4 asks for mass outside
  "diagonal and sink ranges" but not for the sinks' location]
- **H3 specifics.** Periodic one-pixel-frame latent frames (`FRAME_PER_TOKEN`,
  `comfy/ldm/minimax/model.py:30`) [V]; small value norms would mean a routed
  sink costs little in the output even if it takes mass [I, from 2504.10317
  Fig. 9b].
- **Measure first.** Ranked item 1's per-frame incoming-mass curve (every
  latent frame, not pre-binned), plus the fraction of Sol's missed mass
  (measure B(i) in `bench/analyze_sol_block_grouping.py`) that falls on
  whichever frames the curve singles out. Read frame 0 and a mod-5 period as
  separate hypotheses (ranked item 2).

### 2.2 The prefix under INT8: sinks, modality statistics, the anchor (ranked 3)

- **What.** Sinks sit on conditioning tokens in joint-attention DiTs (2605.09313,
  above); LLM KV quantisation protects sink tokens' precision and finds sinks
  beyond the first positions (KVSink, arXiv 2508.04257, COLM 2025, abstract);
  multimodal PTQ finds per-modality smoothing necessary (MASQuant, arXiv
  2603.04800, CVPR 2026, abstract, its "smoothing misalignment"; MorphoQuant,
  arXiv 2606.04349, abstract, "extreme distribution heterogeneity" across
  modalities in an omni model). A search snippet attributed a 10-100x
  visual-over-text activation ratio to MASQuant; not in the abstract text
  returned, so [?].
- **H3 specifics.** [V] One K centre per head for the whole sequence in all
  three INT8 paths: Sol's per-head mean over key blocks
  (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_preprocess.cu:101-115`), kitchen's dense-path anchor from nine
  evenly spaced keys (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/quant_qk_int8.cu:447-453`), sage's global mean
  (`coderef/sage-fork/sageattention/core.py:967`); the forced-exact rows are
  still INT8 in the kitchen lineage (2026-09-17 survey, D.2). Quantisation
  tiles that straddle the text/audio and audio/video boundaries mix modalities
  under one scale; whether any does is derivable from `bench/count_packed_rows.py`
  and the tile sizes in the survey's C.1 table, before touching a capture [I].
- **Measure first.** Ranked item 3. This builds on the survey's C.1 (scale
  granularity) and C.6 (bias correction), which treat the sequence as one
  population; the new axis is the segment.

### 2.3 Drift-gated KV reuse from diffusion LLMs (ranked 5)

- **What.** Elastic-Cache, above (1.4). Related dLLM caching work exists and
  was not opened this session [?].
- **Against `docs/research/technique_transfer.md`.** Agrees with the "KV prefix cache" row
  (n/a for exact reuse, by fact 2) and goes beyond it: approximate reuse under
  fact 2 is what dLLMs do training-free, so the status for the DiT's prefix
  rows is "possible, unmeasured", not "n/a". The payoff is proportional to
  prefix size, so it matters only at video-reference load.
- **Measure first.** Part 1.5 item 2.

### 2.4 Reorder and quantisation (ranked 6)

- **What.** PAROAttention (arXiv 2506.16054, NeurIPS 2025): offline choice
  among the six axis permutations of `[F, H, W]` per layer and head, fixed
  across timesteps and prompts, scored by sparsity plus incoherence
  (max over mean-abs); INT8/INT4 Q, K and P in 64x64 blocks; training-free;
  removing the reorder costs PSNR in both its sparse and its quantised
  configurations (Table 3). CogVideoX-5B, Wan2.1-14B, FLUX.1-dev. [V, HTML
  digest] `docs/morton.md` cites it "from memory, not read in source"; this
  is the reading.
- **Transfer.** Partial, ranked item 6: P path no (row-max u8 in kitchen), K
  path possibly. Note also PARO's per-head permutation is a per-head policy
  the node cannot set (question review, section D). [I]
- **Measure first.** INT8-K-alone error, raster against `3d`, on set A.

### 2.5 Audio-video synchronisation structure (ranked 7)

- **What found.** Temporal-aligned RoPE is the common mechanism: JoVA borrows
  it from MMAudio for joint self-attention (arXiv 2512.13677, section 3.2);
  LTX-2 applies temporal RoPE inside dual-stream cross-attention (arXiv
  2601.03233, abstract); In-Context Audio Control masks its 3D attention to
  enforce alignment (arXiv 2512.18772, abstract). No per-head analysis of a
  single-stream joint audio-video sequence found [?; searches listed under
  ranked item 7].
- **H3 specifics.** [V] Audio rows advance along the same time axis as video
  from the same origin, and the two stereo channels sit at the frame grid's
  `w` extremes with `h` zero (`comfy/ldm/minimax/model.py:117-123`, called at
  `:440-442`). Video advances `FRAME_RESCALE` = 5/3 units per pixel frame
  (`:31`, `:95-96`); audio advances 1.0 unit per latent frame (`:120`), and
  `temporal_shape(362)` gives `audio_t` 603 (`docs/SOLATTN.md`, the recomputed
  sink table), which is 5/3 audio latents per pixel frame. Both streams
  therefore move 5/3 units per pixel frame: H3 already has temporal-aligned
  RoPE in the JoVA/MMAudio sense [I, arithmetic from two constants and one
  recorded length; no frame rate assumed]. A head that attends from audio to
  the left or right image edge would be using the stereo prior [I].
- **Measure first.** Ranked item 1's cross-modal columns with a time-distance
  histogram. If sync heads exist and are few, C.5's queued audio-row error
  should be reported per head rather than pooled.

### 2.6 Test-time registers for outlier tokens (not ranked)

- **What.** High-norm outlier tokens in DiTs "attract disproportionate
  attention while carrying limited local information"; a training-free
  "recursive test-time registers" variant exists (arXiv 2605.05206,
  abstract; RAE-DiT image models). Massive activations in fixed channels
  drive local detail in SD3 and FLUX (arXiv 2510.11538, abstract). [V
  abstract-level]
- **Why not ranked.** Registers change the model's function, not a kernel,
  and the loud-K-channel work already covers the channel side (2026-09-17
  survey, C.5). If ranked item 1's sink map finds high-norm video keys taking
  mass, reopen. [I]

### 2.7 Draft-and-verify sampling (not ranked)

- **What.** Draft-and-Target Sampling drafts a coarse trajectory with large
  steps and verifies it with small steps of the same model, training-free, up
  to 2.1x on robot-policy video models (arXiv 2603.13438, abstract). [V
  abstract-level]
- **Against `docs/research/technique_transfer.md`.** Its speculative-decoding row says the
  diffusion analogue needs a trained draft; this one uses the same model, so
  "trained" is not required. The row's other condition still holds: at the
  distilled checkpoint's step count there is little slack, and the step lane
  sits next to a declined one (`docs/roadmap.md`, via `docs/research/technique_transfer.md`).
  Leave the row's status as n/a for this box. [I]

---

## Against `docs/research/technique_transfer.md`, row by row

- **Fact 2** ("a reference row's K and V at block 1 already depend on the
  video rows"): agrees, and sharpens it. Block 0's reference K/V are
  step-invariant (1.4); text rows are not, because their modulation follows
  `t_v`. [V code]
- **KV prefix cache, "n/a for the DiT"**: agrees for exact reuse. Beyond it:
  (a) the 2026 caching systems all trained one-way attention, which is the
  TaoMate pattern applied to references; (b) dLLMs show drift-gated approximate
  reuse under fact 2, so "possible, unmeasured" for the prefix rows (2.3).
- **Speculative decoding, "n/a"**: status unchanged; the "trained draft"
  premise is weaker than stated (2.7).
- **Token merging and pruning, "possible, unmeasured"**: a new case the row
  does not have. Pruning **reference** rows attacks the forced-exact floor,
  so it stacks with Sol rather than double-counting (1.3).
- **Token reordering (Hilbert, Morton), "exists"**: adds a quantisation
  channel through the K tiles, not the P path (2.4).

---

## Sources

Papers (each opened on arXiv this session; "HTML" means read at section level
through a digesting fetch, "abstract" means only the abstract page):

- FullDiT2, arXiv 2506.04213 (HTML): https://arxiv.org/abs/2506.04213
- OminiControl2, arXiv 2503.08280 (abstract, HTML): https://arxiv.org/abs/2503.08280
- LIVEditor-14B / In-context Sparse Attention, arXiv 2605.04569, ICML 2026 (abstract, HTML): https://arxiv.org/abs/2605.04569
- ToPi, Token Pruning for In-Context Generation in DiTs, arXiv 2602.01609 (abstract, HTML): https://arxiv.org/abs/2602.01609
- Vision-Language Binding in In-Context Image Generation, arXiv 2605.24624 (abstract, HTML): https://arxiv.org/abs/2605.24624
- DIAL, Harnessing Intrinsic Subject-Aware Attention for Controllable Multi-Subject Video Generation, arXiv 2609.11507 (HTML): https://arxiv.org/abs/2609.11507
- HALO, Controlling Motion Transfer in Diffusion Transformers via Attention Heads, arXiv 2607.11081 (HTML): https://arxiv.org/abs/2607.11081
- Stand-In, arXiv 2508.07901 (HTML v4): https://arxiv.org/abs/2508.07901
- Avatar V, arXiv 2606.13872 (HTML): https://arxiv.org/abs/2606.13872
- Elastic-Cache, Attention Is All You Need for KV Cache in Diffusion LLMs, arXiv 2510.14973 (abstract): https://arxiv.org/abs/2510.14973
- Analysis of Attention in Video Diffusion Transformers, arXiv 2504.10317 (abstract, HTML): https://arxiv.org/abs/2504.10317
- Attention Sinks in Diffusion Transformers: A Causal Analysis, arXiv 2605.09313 (HTML): https://arxiv.org/abs/2605.09313
- VMonarch, arXiv 2601.22275 (abstract, HTML): https://arxiv.org/abs/2601.22275
- KVSink, arXiv 2508.04257 (abstract): https://arxiv.org/abs/2508.04257
- MASQuant, arXiv 2603.04800 (abstract): https://arxiv.org/abs/2603.04800
- MorphoQuant, arXiv 2606.04349 (abstract): https://arxiv.org/abs/2606.04349
- PAROAttention, arXiv 2506.16054 (abstract, HTML): https://arxiv.org/abs/2506.16054
- JoVA, arXiv 2512.13677 (HTML): https://arxiv.org/abs/2512.13677
- LTX-2, arXiv 2601.03233 (abstract): https://arxiv.org/abs/2601.03233
- In-Context Audio Control of Video Diffusion Transformers, arXiv 2512.18772 (abstract): https://arxiv.org/abs/2512.18772
- Taming Outlier Tokens in Diffusion Transformers, arXiv 2605.05206 (abstract): https://arxiv.org/abs/2605.05206
- Massive Activations are the Key to Local Detail Synthesis in DiTs, arXiv 2510.11538 (abstract): https://arxiv.org/abs/2510.11538
- Draft-and-Target Sampling for Video Generation Policy, arXiv 2603.13438 (abstract): https://arxiv.org/abs/2603.13438

Code read (ComfyUI core and this repo cited by repo-relative path; clones as
`coderef/<clone>/<path>`):

- `comfy/ldm/minimax/model.py` (ComfyUI core): `:5-7`, `:30-33`, `:95-96`,
  `:117-123`, `:181-199`, `:291-297`, `:350-470`, `:532-545`, `:633-637`.
- `sol_attn_h3.py:430` (`_perm_for`), `:894-946` (`_sink_blocks`).
- `workflows/h3_config.py`, `SOL_CORE_DEFAULTS`, `SOL_RECOMMENDED`.
- `docs/wiki/decisions.md:51`; `docs/SOLATTN.md`, "References are pinned
  exact" and the sink table; `docs/morton.md`, "Related papers".
- `coderef/comfy-kitchen/comfy_kitchen/__init__.py:175-177`.
- `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_preprocess.cu:101-115`, `:251-267`.
- `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/quant_qk_int8.cu:447-453`.
- `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/attn_utils.cuh:36-40`.
- `coderef/sage-fork/sageattention/core.py:967`.

Not opened, cited only as absent or unverified: the sglang and LightX2V H3
reference-row sink handling; dLLM-Cache and Fast-dLLM; DFSAttn's Hilbert
reorder (already listed in the 2026-09-17 survey, D.9); EditYourself
(arXiv 2601.22127) and the other 2026 talking-avatar papers surfaced by search.
