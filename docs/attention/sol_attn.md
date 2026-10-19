# Sol-Attn on MiniMax H3

Sol-Attn ([arXiv 2607.24027](https://arxiv.org/abs/2607.24027)) is training-free block-sparse attention. The packed
sequence is cut into 64-token blocks. Each query block attends a routed subset of key blocks exactly and covers every
other key block with one pooled term, so the whole sequence still contributes to the softmax denominator.

**This page is the Sol-Attn authority.** It states what Sol-Attn is, what ships, what each knob does and what has been
measured about it, and what not to rely on. The detail is on the pages below; a number is stated once, on the page
that owns it, and if this page and a detail page disagree about something the detail page owns, the detail page is
right.

| page | what it owns |
|---|---|
| [`sol_kernel.md`](sol_kernel.md) | the kernel: building and installing it, checking it runs, what it exposes and what the node does with each argument |
| [`sol_wiring.md`](sol_wiring.md) | where the node sits in a graph, and the composition traps that make Sol silently not run |
| [`sol_sink.md`](sol_sink.md) | the conditioning sink: which rows are forced exact, what references cost, the sink modes |
| [`sol_dense_blocks.md`](sol_dense_blocks.md) | which blocks to keep off Sol: every ranking measured, and how the shipped list was chosen |
| [`sol_error.md`](sol_error.md) | where Sol's error comes from: sparsity against the kernel's INT8, by depth, and what `tau` does to it |
| [`sol_measured_arms.md`](sol_measured_arms.md) | the timed arms and the quality judgements, with the conditions each was taken under |
| [`sol_instruments.md`](sol_instruments.md) | the route observer, the call timer, the tau sweep, the commands that reproduce a number, and the traps hit making them |
| [`sol_upstream.md`](sol_upstream.md) | what upstream ships and claims, and what we would adopt from it |
| [`morton.md`](morton.md) | token ordering, a closed lane |

Canvas and frame count against Sol's 64-token blocks (where the token floor falls, which lengths and canvases tile
cleanly) are in [`../h3/geometry.md`](../h3/geometry.md), with the tables in [`../h3/geometry_cost.md`](../h3/geometry_cost.md), "The Sol-Attn token floor". Repo-wide ledgers sit above all of this:
[`../method/evidence.md`](../method/evidence.md) for measured claims not to rely on,
[`../method/checks.md`](../method/checks.md) for what is guarded, and
[`../state/open_experiments.md`](../state/open_experiments.md) for what is deliberately not measured.

## What the published method claims

The paper was read at the depth recorded in [`sol_upstream.md`](sol_upstream.md) (abstract, ablation summary and
HTML, not the full PDF). Four things shape how everything here is read:

- **The contribution is the correction, not the routing.** Unselected blocks contribute their pooled proxy score
  instead of being dropped, and the paper's ablation shows that advantage widening as sparsity rises. So the pooled
  tail is the method; the node has no switch for it.
- **`tau` is the paper's beta** in `t_i = mu_i + beta * sigma_i`, and the paper never sweeps it.
- **H3 is not evaluated in the paper.** The sm89 build, the H3 port and everything on these pages sit on top of the
  published method.
- **No token reordering appears in it** ([`morton.md`](morton.md)).

Everything here is one machine (RTX 4090, sm_89) and one model. Ratios against sage are against the
[SageAttention-ada](https://github.com/fblissjr/SageAttention-ada) fork, never against stock SageAttention.

## What ships

Read the values from the code; the prose below says where they live and why, not what they are.

- **The node** is `MiniMaxH3Sol` ("MiniMax H3 Sparse Attention", `sol_attn_h3.py`), redesigned on 2026-09-27
  ([`../research/2026-09-27_sol_node_redesign.md`](../research/2026-09-27_sol_node_redesign.md)). Its
  `define_schema` defaults are the recipe.
- **The recipe** is `workflows/h3_config.py::SOL_RECOMMENDED_CUDA`. `sol_for_graph(pdd, steps, distill_lora)` is the
  one resolver the generator and the checks both call; the PDD and FlashGen overrides
  (`SOL_PDD_OVERRIDES`, `SOL_DISTILL_LORA_OVERRIDES`) and the per-step `end_percent` table
  (`SOL_END_PERCENT_BY_STEPS`) are empty, so every graph carries the same recipe. `SOL_CUDA_DEFAULTS` mirrors the
  node's own defaults, and `bench/check_sol_kernel.py` grades it against the schema.
- **On by default in every shipped video workflow.** The graphs that legitimately ship without it are
  `bench/check_attention_defaults.py::SOL_EXEMPT_STEMS`, each with its reason.
- **The dense kernel under Sol** (the steps outside the sigma window, the blocks in `dense_blocks`, calls below
  `min_tokens`) is core's Model Attention Backend on kitchen's rotated `int8_attention`
  (`h3_config.DENSE_BACKEND_NODE`), since 2026-09-15. The graphs that keep sage or stock attention there instead are
  `bench/check_attention_defaults.py::FLOOR_STEMS`. Wiring and order: [`sol_wiring.md`](sol_wiring.md).
- **The kernel** is `comfy_kitchen.sol_attn`, a local build of the owner's kitchen fork carrying the commits upstream
  does not (`blk_cnt`, `qk_balance`, `rotate`). `bench/check_sol_kernel.py --require` says which build is importable
  and whether it is usable. [`sol_kernel.md`](sol_kernel.md).
- **Lane overrides** live with the lane: the masked video-to-video lane sets its own sink
  (`h3_config.MASKED_JOB_SOL_SINK`, the owner's 2026-10-10 decision for that lane), which does not move the node
  default.

## The knobs, and what is measured about each

Inputs in widget order, which is not cosmetic: an editor-saved graph stores `widgets_values` as a bare list matched by
index. `token_routing` and `rows` are DynamicCombos whose sub-inputs are keyed `token_routing.blocks`,
`rows.text_rows` and so on in API form; `workflows/build_workflows.py` owns both spellings, so regenerate rather than
hand-edit. Shipped values are the constant named, read 2026-10-19.

| knob | shipped | what it does | what is measured | detail |
|---|---|---|---|---|
| `tau` | `SOL_RECOMMENDED_CUDA["tau"]` (1.0 since 2026-08-20, owner) | routing threshold in sigmas of the proxy row; a key block is exact when its mean score clears `tau * sqrt(var)`. Higher is sparser. Upstream densities: 1.0 keeps about 16% exact, 1.5 about 7%, 2.0 about 2.7% | 1.0 has lower sparsity error than 1.3 on one capture, and is the lowest value graded; above about 1.5 a small object can dissolve mid-clip. The full sweep instrument exists and has not fitted a table yet | [`sol_error.md`](sol_error.md), [`sol_measured_arms.md`](sol_measured_arms.md) |
| `quantizer` | `rotated` since 2026-09-27 (owner) | how Sol's INT8 quantizers treat q and k: `plain`, `balanced` (`qk_balance`), `rotated` (`rotate`, a Hadamard rotation before INT8), `balanced+rotated` | on captures of the shipped PDD8 graphs `rotated` has the lowest quantization error on every graded cell and costs less than `balanced`; Sol's total error moves little because sparsity dominates (`bench/results/2026-09-27_sol_redesign_test2.md`) | [`block49_quant_error.md`](block49_quant_error.md), [`quant_policy.md`](quant_policy.md) |
| `dense_blocks` | `SOL_DENSE_TAIL` (`"38,39,40,41,42,49"` since 2026-10-02, owner) | blocks handed to the dense kernel under Sol instead of Sol. Not exact attention: the fallback is kitchen INT8. Negative indices count from the end (`block_spec.py`) | chosen on local probe error and kept by decision once the output-level panel found the list does not move the verdict (`bench/results/2026-10-02_sol_dense_blocks_panel.md`) | [`sol_dense_blocks.md`](sol_dense_blocks.md) |
| `sink_conditioning` | `SOL_SINK_DEFAULT` (`exact_kv_and_all_rows` since 2026-10-02, owner) | which conditioning rows are exact: `exact_kv`, `exact_kv_and_rows` (target-audio queries exact too), `exact_kv_and_all_rows` (every conditioning query exact), `off`. Modes are `sol_attn_h3.py::SINK_CONDITIONING_MODES` | `all_rows` lost on no scene of the 2026-10-02 panel and was the one Sol arm rated fine on dialogue; on a video reference it runs most of the reference rows exact, which is most of Sol's saving | [`sol_sink.md`](sol_sink.md) |
| `token_routing` | `off` | per-block extra individual-token routing (kitchen's `token_aug`), budget 64 in every preset: `off`, measured blocks, early and middle, all blocks (needs a balanced quantizer), `custom` | lowered error on four captured blocks and raised it on block 49; **no render has been judged with any preset on**. Asserted by `bench/check_token_routing.py` | [`../research/2026-09-04_sol_token_aug_grade.md`](../research/2026-09-04_sol_token_aug_grade.md) |
| `start_percent` | 0.2 on every graph since 2026-10-05 (owner) | dense before this point of the sigma window | 0.0 on the PDD8 ref2va arms lost the subject's video reference (`bench/results/2026-10-05_masked_v2v_motion_arms.md`); on the t2v PDD8-to-FlashGen finish the owner could not tell 0.0 from 0.2 blind and 0.0 was faster (`bench/results/2026-10-01_start_percent_panel.md`). Unmeasured on the base graphs. On the 16-step graph the dense steps before it cost more attention time than all the Sol steps | [`sol_measured_arms.md`](sol_measured_arms.md), "The sigma window is not a step fraction" |
| `end_percent` | 1.0 since 2026-09-11 | dense after this point | adopted because sglang's backend and core's node agree on running Sol through the last step; never measured at either value | [`sol_upstream.md`](sol_upstream.md) |
| `min_tokens` | 12288 since 2026-08-27 | shorter calls fall through to the dense kernel under Sol | unmeasured; every DiT call here is 31k-128k tokens and every refiner call about 311 rows, so 4096 and 12288 select identically on everything this repo renders | [`sol_wiring.md`](sol_wiring.md) |
| `verbose` | True since 2026-09-17 (owner) | one log line per distinct shape per process: sparse or dense and why, the options, the sink ranges. Separately, one `[h3-sol] on:` line per patch | a render where Sol silently stayed dense looks exactly like one where it ran | [`sol_kernel.md`](sol_kernel.md), "Verify it" |
| `tau_table` | none | a file in `sparse_tables/` with a tau per head of the blocks it lists (`sparse_table.py` owns the format) | no table ships; nothing judged with one | [`sol_instruments.md`](sol_instruments.md) |
| `rows` | follows `sink_conditioning` | `per segment` sets text, reference and target-audio query rows exact or routed apart, through the kernel's `tau_map` | nothing judged or measured at the output | [`sol_sink.md`](sol_sink.md) |

**Retired 2026-09-27, with the code only they reached:** `selection` and `keep_percent` (top-k; the SLA lane closed),
`pooled_tail` (always on: it is the method), `morton` and `morton_curve` (Morton closed, not refuted), `tau_profile`
(never wired) and `token_aug_blocks` (now `token_routing`'s `custom`). Their rows are in git before that date.

## Where the gains are

- **Sol-Attn is worth 1.896x on the whole sampler** on this box: 860.8 s dense against 454.0 s sparse, the same
  shipped config with only `dense_blocks` changed (2026-08-16, 1344x768, 294 frames, 16 steps, one run per arm, on
  the kernel and recipe of that date). It is the clean Sol-against-dense number;
  [`sol_measured_arms.md`](sol_measured_arms.md) has it and every other timed arm.
- **A Sol call's time is its exact stage, and the exact stage follows routed density.** Routing and the pooled tail
  are negligible; compute, not staging, is the critical path. The levers are `start_percent` and routed density
  (`bench/results/2026-09-17_sol_stage_profile.md`).
- **The token floor.** Below about 60,000 video tokens a Sol arm returns a null that reads as "this knob does
  nothing"; `bench/bench_e2e_h3.py` warns below it. It is a token floor, not a frame floor, so canvas and length both
  move it ([`../h3/geometry_cost.md`](../h3/geometry_cost.md), "The Sol-Attn token floor").
- **Kernel speed is not end-to-end speed.** Attention is part of a step and Sol runs only inside its window; never
  quote a `bench_minimax_attn.py` number against a `bench_e2e_h3.py` one
  ([`sol_measured_arms.md`](sol_measured_arms.md), "The unit trap").
- **Inside Sol, sparsity costs three to five times what the kernel's INT8 costs through the trunk, and the two are
  equal at block 49** on unrotated INT8 ([`sol_error.md`](sol_error.md)). The INT8 half at the last blocks is what
  the `rotated` quantizer addresses.

## Do not rely on

Each row is a figure or claim that was measured or written here and does not hold for what ships. The ledger is
[`../method/evidence.md`](../method/evidence.md); these are the ones a reader of this folder meets.

| claim | why not |
|---|---|
| every e2e timing from the 2026-08-14 run, including **1.611x** | taken against an **fp8** sage baseline the graphs did not ship, and before it was known that sage runs 5 of 16 steps in the Sol arm. Quote 1.896x instead |
| `centroid_tail` = **2.5%**, or upstream's **5-10%** e2e | the toggle no longer exists: kitchen#117 made the centroid tail unconditional. The A/B closed unanswered |
| `reuse_qkv_memory` "no VRAM saving" | **uninformative, not negative**: the peak-VRAM column reported torch-active bytes, which could not have shown a saving. The input is gone |
| any fp8-vs-fp16 sage accuracy ratio from before 2026-08-18 | withdrawn on 2026-08-16 and deleted, not caveated: a `torch.randn` sweep, and a secondhand figure never re-derived here. The capture-based grade that replaced it is `data/results/2026-08-18_sage_accuracy_on_capture.json` ([`sol_measured_arms.md`](sol_measured_arms.md), "Quality") |
| **0.999919** and the other correctness cosines | **implementation fidelity, not accuracy**: the kernel is graded against the reference at the same tau, so the sparse approximation cancels. T=512, synthetic |
| **1.4x** CUDA over Triton e2e | upstream conversation, never reproduced here, and the Triton pack is deleted |
| "38 text rows", "sequence 12,264" | from `smoke_h3.py`, which substitutes both the prompt and the length. Everything derived from it (that audio dominates the sink, that `sink_q` starts at 0) describes the harness. The shipped t2v graph has 311 text rows |
| the reference-load table's **57.9%** as the price of `exact_kv_and_rows`, and the "23-point swing" between it and `exact_kv` | that formula runs every conditioning query exact, which `exact_kv_and_rows` does not: between those two modes the swing is about 0.5 points. The same figure is the price of `exact_kv_and_all_rows`, the shipped mode since 2026-10-02 ([`sol_sink.md`](sol_sink.md)) |
| "with Sol on, sage gets nothing" | **retracted**: the dense kernel under Sol runs every step outside the sigma window, 5 of 16 at the old 0.2/0.9 window, not only the calls under `min_tokens` |
| H3's DiT has "exactly one attention site" | true of the source line, misleading about the modules: one line reaches 52 attention modules, the 50 DiT blocks at full length and 2 token-refiner blocks at the text span. The VAE's attention passes no `transformer_options`, so no override reaches it |
| Morton "worth 1.16x alone, 94% GPU utilisation" | **retracted 2026-08-16**: Triton, 362 frames, stacked on int8. On the CUDA kernel the permutation is free ([`morton.md`](morton.md)) |
| any quality A/B from before 2026-08-15 as a like-for-like | taken on `res_multistep`; the sampler has injected noise every step since, so a numerics change reads as more of a reseed. Timings carry, quality comparisons do not |
| any figure quoted to more than three significant figures across the 2026-08-29 kernel merge | the merged kernel moves a Sol arm by rel L2 7.67e-05 (`data/results/2026-08-29_sol_kernel_merge_equivalence.json`); figures survive at three |
| a bench's progress read from its own stdout mid-run | the warmup `print` lacks `flush=True`, so under `tee` a finished render can read as still warming up for twenty minutes. Read ComfyUI's log |

"int8" means three things in these records, and a claim that says "int8" without saying which cannot be carried:

| it may mean | how to tell |
|---|---|
| the attention kernel: Sol routes and computes in INT8 unconditionally | `cuda-int8` in the log |
| the DiT and encoder weights, `pruned_int8_convrot` in `h3_config.MODELS` | nothing in a log: the loader prints `dtype: torch.float16` for both builds |
| the VAE decoder, a separate `int8_convrot` build | resident allocation, not a log line |

The weights are pruned, convrot-rotated and int8, three separate properties, so anything here that reasons about q/k
geometry reasons about a rotated, pruned, quantized space.

## Can rely on

- **The seam works**, and a failure is silent: verify it on every new kernel build ([`sol_kernel.md`](sol_kernel.md)).
- **362 is the longest window and 345 is where diffusers stops**: `h3_rules.MAX_LENGTH` and
  `h3_rules.reference_would_emit()` answer two different questions.
- **The CUDA and Triton backends were arithmetically equivalent** at T=512 fidelity, each graded in its own tail mode.
- **The node dispatch is bitwise identical to the kernel call it should make**: `bench/check_sol_node_equivalence.py`.

## What is open

| question | why it matters | blocker |
|---|---|---|
| `tau` beyond the tested points, and a tau per head | 1.0 is the lowest value graded and errors were still falling | the tau sweep is built; a table has to be fitted and graded ([`sol_instruments.md`](sol_instruments.md)) |
| `start_percent` on the base graphs | its dense steps cost more attention than all Sol steps on the 16-step graph | a blind panel; the arms exist |
| which `dense_blocks`, judged at the output on more than one seed | the shipped list rests on local probe error and a panel stopped after one judged seed | GPU hours ([`sol_dense_blocks.md`](sol_dense_blocks.md)) |
| `min_tokens` against the kitchen dense kernel | the crossover was argued, never measured | a sweep at short lengths |
| ours as shipped against core's `BlockSparseAttention` | the 2026-09-10 parity grade predates `rotated`, the kitchen fallback and the dense list | a capture grade (`bench/results/2026-09-27_attention_parity.md`) |
| kitchen's sm89 kernel against NVLabs' `cute_sm89` | the only external cross-check on this card | a seam: their API has no `sink_q` ([`sol_upstream.md`](sol_upstream.md)) |
| per-step tau, and text and audio exact with references routed | the two upstream H3 policies we do not express | graded on captures first ([`sol_upstream.md`](sol_upstream.md), "What we would adopt") |
| quality at the shipped tau, watched to the end | the object-dissolve artifact is temporal, and stills cannot show it | a person watching |
