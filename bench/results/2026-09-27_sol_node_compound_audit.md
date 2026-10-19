# MiniMaxH3SolAttn against core's Model Sparse Attention, as one system (2026-09-27)

The owner asked five things:
- what every knob in our Sol node and in core's `BlockSparseAttention`
  ("Model Sparse Attention") does;
- how those knobs combine, taken as one system;
- where the two nodes differ under the hood, and why;
- which of our knobs do nothing or should be removed, consolidated or
  changed;
- whether any upstream repo does it our way.

This is a read-only code audit by the lookingdude session and four
subagents. Nothing new is measured. Every number below is quoted from the
record it cites, and every file:line citation is as of today. It is the
sibling of [`2026-09-27_attention_parity.md`](2026-09-27_attention_parity.md),
which holds the node-against-node comparison; that comparison is not
repeated here.

**Sources as read today:**
- **ours:** `sol_attn_h3.py`, `workflows/h3_config.py`;
- **core:** ComfyUI's `comfy_extras/nodes_sparse_attention.py`, in the
  checkout two levels above this pack, at 4ef23c34 (unchanged since
  00d34d92, 2026-09-08);
- **kitchen:** the Sol sources in `coderef/comfy-kitchen`, whose line
  numbers are from the `h3-frontier` worktree.

The installed kitchen changed during the audit, from `0.2.35+sol.8176242` to
`0.2.35+sol.863e953.up.c8c7825` (CHANGELOG 0.158.0). The Sol code is
identical in both builds; the carried diff matches apart from hunk offsets.
Only the dense `int8_attention` differs (§2.3).

---

## 1. What actually runs, shipped configuration

### Our node

Each attention call passes through the override (`sol_attn_h3.py` (lines 1122-1208 as of 2026-09-27; the file was restructured 2026-09-27))
in this order:

```
mask present                          -> dense
block in dense_blocks (45,48,49)      -> dense
sigma outside [end, start] (0.2..1.0) -> dense
not CUDA / not bf16 / head_dim != 128 / cross-attn / T < min_tokens (12288)
                                      -> dense
else -> comfy_kitchen.sol_attn(q, k, v,           # direct entry, full Q/K/V
          tau=1.0, tail=True,
          sink_blocks=[0, ceil(video_start/64)],  # exact_kv_and_rows
          sink_q=[audio_start/64, same end],
          token_aug=0,                            # token routing off
          qk_balance=True, rotate=False)
kernel exception                      -> dense (logged)
```

"Dense" means `previous`, the override that was on the hook when Sol
patched.
- In 124 of the 131 shipped Sol nodes (119 graphs, read with json over
  `h3_config.graph_paths`), that is core's `ModelAttentionBackend` "comfy
  kitchen attention", so the call runs `comfy_kitchen.int8_attention`
  (§2.3).
- The other 7 are 5 sage arms and 2 Sol-only graphs.

So on a shipped 345-frame render:
- the first 20% of steps run all 50 blocks on kitchen int8;
- after that, blocks 45, 48 and 49 run on kitchen int8 and the other 47 on
  Sol;
- the short text-refiner calls run dense.

### Core's node, at its defaults, on H3

- **H3 blocks:** each block gets a `double_block` patch that calls
  `sol_attn_chunked` (core:248-307).
  - It takes qkv in 4096-row chunks straight from `qkv_proj` and runs
    RMSNorm and RoPE in-kernel, so Q/K/V are never built.
  - Settings: tau 1.3; tail True (the default, which core never passes);
    `extra_tokens` 256 on every block; `sink_conditioning`
    exact_kv_and_rows, the same formula as ours.
  - Statistics carried from the last step: K is centred by last step's K
    mean, and V is quantized with last step's |max| x 1.1. Both are kept per
    (block, padded length, conditioning uuids) (core:263).
  - The first step runs the producer twice.
- **Non-H3 models:** the attention override runs direct `sol_attn`, casting
  fp32 to bf16 (core:204).
- **Dense fallback:** the same idea as ours. Core falls back to whatever
  override was under it, or `func`, before start 0.2, on `dense_blocks`
  (empty by default), and below 12288 tokens.
- **Re-installs its override on top at every step** (ON_PREPARE_STATE,
  core:337).

---

## 2. The kernel, stage by stage

Taken from the kernel subagent's read of
`comfy_kitchen/backends/cuda/sage_attention/`. lookingdude re-read the lines
behind the §5.1 bug.

### 2.1 One direct `sol_attn` call

1. **Pool.** Block means of K and block sums of V, over live rows
   (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_preprocess.cu` (lines 65-96 at `8176242`)).
2. **K statistics.** `kmean` is the mean of the block means, with every
   block weighted equally. `kcvar[d]` is the variance of the block means
   across blocks (`:101-116`).
3. **Threshold.** thr = tau · sqrt(Σ_d c_d² · kcvar_d) · scale·log2e, where
   c is the query block's centroid (`:57-60`).
   - "tau sigmas": a block's proxy score c·(k̄_b − k̄) has mean zero over b
     and variance cᵀΣc. The code keeps only Σ's diagonal.
   - The threshold is always computed in raw space: the diagonal
     approximation survives a per-channel rescale but not a rotation.
4. **Quantize.** Q rows and centred K rows go to int8 with one scale per
   token. `qk_balance` first scales q·f and k/f. `rotate` applies
   sign·H128/√128. The pooled keys and the centroid get the same transforms.
5. **Route** (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_route.cu` (lines 157-163 at `8176242`)). Block b is exact for query block
   q if its score ≥ thr, or |q − b| ≤ 1, or b is a sink. For `sink_q` rows,
   every live block is exact.
6. **Tail** (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_route.cu` (lines 197-276 at `8176242`)). Each unrouted block becomes one pooled
   term: its logit is the centroid score, its value the block's V sum.
   `tail=False` drops it.
7. **Token stage** (only when `token_aug` > 0; `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu`).
   - Query blocks are paired (TOK_GROUP=2) and a group centroid is formed.
   - For blocks that both partners left unrouted, every token is scored and
     the top ones are admitted, by histogram bins, up to the budget.
   - Admitted tokens go exact. The rest are scored token by token at the
     centroid, and that replaces those blocks' pooled terms.
8. **Exact.** int8 QK, then u8 P · int8 V, merged with the route's
   (o, m, l).
9. **Coarse** (VSA only, in Python). Dense attention over the block means,
   times the gate, added to each token.

### 2.2 Arguments that are inert in a given configuration

| Argument | Inert when | Decided at |
|---|---|---|
| `tau` | `topk_ratio > 0` | `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn.cu` (line 161 at `8176242`) |
| `tau`, `topk_ratio`, tail | query rows inside `sink_q` | `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_route.cu` (line 163 at `8176242`) |
| `token_aug` | `sink_q` rows, **and their pair partner**: a non-sink block paired with a sink_q block loses token routing | `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_route.cu` (lines 186-189 at `8176242`) |
| `token_aug`'s remainder | `tail=False` (selection still runs) | `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu` (line 340 at `8176242`) |
| `qk_balance` | heads whose top-4 K channels hold < 20% of K energy; the codes are bit-identical | `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_preprocess.cu` (lines 184-188 at `8176242`) |
| `qk_balance`, `rotate` | the threshold and the coarse branch, which are always raw space | `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_preprocess.cu` (lines 327-332 at `8176242`) |
| `scale` | the tau routing decision (thr scales with it) | `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_preprocess.cu` (lines 58-60 at `8176242`) |
| `blk_cnt` | always: it is output only | `cuda/__init__.py` |

### 2.3 The dense path: `int8_attention`, where "dense" goes

- **Q:** int8, one scale per 4 rows ("per-thread").
- **K:** int8, one scale per 32 keys, with a representative-key shift
  (`detect_k_anchor`: a sampled key, not a mean).
- **Rotation:** q and k get the same sign·H128 rotation as Sol's `rotate`
  (`convrot128`, the same four sign words).
- **V and PV:** V is int8 per channel. PV is u8·s8 with int32 accumulation;
  softmax is fp32.
- **What the new build changes** (#208, on sm_89, CTA 128/128, no mask):
  - Q stays in registers.
  - The launcher uses CTA_Q=64 when 4096 ≤ qo, kv ≤ 16896.
  - The quantizers are file-identical across the two builds, so these are
    scheduling changes. That is the subagent's inference from reading.
  - fastdude's after-run measures it: `2026-09-27_int8attn_after.jsonl` and
    `2026-09-27_pdd_strength_after.jsonl`.

---

## 3. Every input of our node, and its counterpart in core

"Shipped" is `h3_config.SOL_RECOMMENDED_CUDA` / `SOL_CUDA_DEFAULTS`.
"Off-default" counts nodes out of the 131 shipped Sol nodes.

| Our input | Default / shipped | What it does | Off-default | Core's counterpart |
|---|---|---|---|---|
| `selection`: adaptive tau, `tau` | 1.0 / 1.0 | §2.1 step 3 | 0 | `sol-attn`, `tau` 1.3 |
| `tau_profile` (socket) | unset / unset | per-block tau override | **never wired** | none |
| `selection`: top-k, `keep_percent` | 10 / not shipped | a fixed fraction of key blocks; tau is ignored | none of the 131 use it | `sla`, `keep_percent` 10, tail always on |
| `pooled_tail` | True / True | sets `tail=` | 0 | none: always True for sol-attn and sla, False for vsa |
| `start_percent`, `end_percent` | 0.2, 1.0 / same | sigma window | 2 graphs (`pdd8_sol_narrow`, `sol_only`) | same, with the same defaults |
| `min_tokens` | 12288 / same | shorter calls go dense | 0 | same |
| `sink_conditioning` | exact_kv_and_rows / same | §2.1 steps 5 and 6 | 1 (`sol_allrows`: `exact_kv_and_all_rows`) | same, without `exact_kv_and_all_rows` |
| `dense_blocks` | "45,48,49" / same | the block leaves Sol | 3 set it to "" | same input, default "" |
| `qk_balance` | **False / True** | §2.1 step 4 | True in 129: the node default is the off-default value almost everywhere | none; its chunked entry has no such argument |
| `rotate` | False / False | §2.1 step 4 | 2 | none |
| `token_routing` + `token_aug_blocks` | off, "" / same | §2.1 step 7, budget 64 per block | 0 | `extra_tokens` 256, global |
| `morton`, `morton_curve` | False, 3d / same | Z-orders the video rows before blocking | 0 | none. Core's merged node never had it: no commit since e308cc73 has it. It came from kijai's pre-merge node, vendored as `vendor/sol_attn_minimax.py` |
| `verbose` | True / True | logging | 0 | `verbose`, default False |
| (none) | | | | `sla` and `vsa` in the same node |

---

## 4. The compound system: how the knobs combine

1. **`dense_blocks` overrides every Sol knob on those blocks.**
   - On 45, 48 and 49, tau, `tau_profile`, `qk_balance`, `rotate`, token
     routing, the tail and the sink all do nothing. What runs there is the
     dense fallback: kitchen int8, with its own rotation.
   - `qk_balance`'s gate is recorded as opening on exactly those blocks:
     "no per-block list: blocks 45, 48 and 49 open on their own"
     (`docs/attention/block49_quant_error.md` line 543 at `901bee1`). Blocks 0, 32 and 40 measured
     below the gate (`2026-09-15_channel_balance_kernel_b{0,32}_s15.json`).
   - **So with the shipped dense tail, shipped `qk_balance=True` is probably
     a no-op.** This is inferred, not measured: blocks 1-44, 46 and 47 were
     never captured. §6 gives the bit-identity test that settles it.
   - **Refuted, 2026-09-27 (test 1 of the redesign).** Balanced and plain
     render differently in every mode, and the gate opens on some heads of
     every captured block (`2026-09-27_sol_redesign_test1.md`). The two
     sources above did not say what this bullet read into them. The
     block-49 page's own table moves blocks 0 and 40 at the default gate,
     and the b0 record's `kernel` row differs from its `plain` row. Blocks
     44 to 48 had also been captured, on 2026-09-19.
2. **The sigma window runs before everything.** Outside it, every block is
   dense. Morton still permutes every forward regardless (`:633-640`). That
   is invisible to exact attention and changes only int8 rounding.
3. **Sink, tail and token routing are tied together.**
   - Sink KV blocks are never tail terms or token candidates.
   - `sink_q` rows have no tail and no routing at all.
   - Under `exact_kv_and_rows`, the query block paired with the first audio
     `sink_q` block also loses token routing (§2.2).
   - With `pooled_tail=False`, token routing keeps its selected tokens and
     drops the rest.
4. **`rotate` and `qk_balance` compose:** balance first, then rotation. On
   the block-49 capture grade (`2026-09-15_sol_token_aug_x_options_b49_s15.json`,
   `fixed_wheel` rows, total L2 against exact), rotation does most of what
   balance does:

   | Arm | L2 |
   |---|---|
   | plain | 0.0415 |
   | balance | 0.0375 |
   | rotate | 0.0349 |
   | both | 0.0347 |

5. **Token routing depends on the quantizer.** From the same record, with a
   64-token budget:

   | Arm | without | with |
   |---|---|---|
   | plain | 0.0415 | 0.0539 (worse) |
   | balance | 0.0375 | 0.0366 (better) |
   | rotate | 0.0349 | 0.0489 (worse) |
   | both | 0.0347 | 0.0337 (the best of any arm) |

   - Our "all blocks" preset demands balance **and** rotate
     (`sol_attn_h3.py` (lines 298-302 as of 2026-09-27; the file was restructured 2026-09-27)), and its error text says token routing
     "measured worse" without them. The record supports "needs balance",
     not "needs both".
   - **Every balance-plus-token arm ran on a kernel with a balance/token
     space mismatch (§5.1).**
6. **top-k disables tau and `tau_profile`.** The tail stays unless
   `pooled_tail` is False, and token routing still applies. The forced
   diagonal adds up to 3 blocks on top of the budget.
7. **Morton does not move the sink.**
   - Only video rows move, and `_sink_blocks` reads the unpermuted video
     span.
   - `_perm_for` puts the ragged remainder in the block it shares with
     conditioning, which the sink keeps exact. With the sink off, that mixed
     block is unprotected.
8. **Node order matters, and ours is fragile.** Sol must be the last
   attention node, and every shipped graph does put it last.
   - A backend node placed after Sol overwrites the override key, and Sol
     stops **silently**, because ours installs only once.
   - Core re-installs on top every step (core:217-226, 337-338), so it
     cannot be displaced this way.
9. **Debug switches.** `H3_SOL_PROBE trajectory=sage` changes the render: it
   returns the fallback's output (`:1197-1207`). `H3_SOL_OBSERVE` (blk_cnt)
   and `H3_SOL_TIME` do not.

---

## 5. Defects and contradictions found

1. **Kernel bug: `qk_balance` plus `token_aug` score in different spaces.**
   - The token stage builds its group centroid from `qmean`, the
     **unbalanced** block mean (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu` (lines 79-84 at `8176242`)). It is rotated
     when `rotate` is on, but never multiplied by the factor.
   - The key rows it scores against are balanced by 1/f (`prep_k` receives
     `fk`).
   - So on every head whose gate is open, token scores are Σ c_d(k_d−m_d)/f_d
     instead of c·(k−m). That distorts token selection, the histogram window
     against `tok_ref` (which the route computes correctly), and the
     remainder's logits.
   - It is the balance twin of the rotate bug that cf18031 fixed, and the
     fork's tests have no qk_balance × token_aug case. The kernel subagent
     found it; lookingdude re-read the centroid kernel and confirmed it.
   - **Reach:** only the direct entry with both options on. Shipped graphs
     run token routing off and are unaffected. It does apply to:
     - the "all blocks" preset;
     - any custom token routing under the shipped `qk_balance=True`;
     - every balanced token arm in the 2026-09-15 grade.
   - **Fix:** in `sol_token_group_kernel`, multiply the group centroid by
     `fq[d]` before the rotation, and add a test that tracks eager. Not done.
2. **Kernel bug, unreachable from either node today.** `sol_attn_chunked`
   with `rotate=True` and `topk_ratio > 0` takes the dot product of the
   rotated workspace `cen8` with unrotated pooled sums
   (`_topk_threshold_from_workspace`). Core never passes `rotate`, and our
   node never uses the chunked entry. It becomes live if either one does.
3. **Possible, unmeasured.** On the chunked entry, exact K is centred by
   last step's mean but the pooled tail by this call's. Within one softmax
   their relative weight then shifts by about q·Δkmean·scale. This is core's
   H3 path, not ours.
4. **Prose that lost to code** (listed for correction, not corrected here):
   - The "all blocks" error text (`sol_attn_h3.py` (line 301 as of 2026-09-27; the file was restructured 2026-09-27)), per §4.5.
   - The `qk_balance` tooltip calls it "an experiment" (`:1693`), yet 129
     graphs ship it on.
   - `docs/state/roadmap.md` lines 757-763 at `901bee1` says Sol "ships off" and that
     `dense_blocks` ships empty.
   - `sol_attn_h3.py:780, 861, 879, 104` say the kernel is bf16-only. Direct
     `sol_attn` takes fp16; only the chunked entry is bf16-only.
   - `workflows/h3_config.py` (line 728 as of 2026-09-27, before the redesign) says empty is how the node spells off. Since
     2026-09-25 it has been `token_routing="off"`.
   - `workflows/h3_config.py` (lines 209-213 as of 2026-09-27, before the redesign) calls kijai "the algorithm's author". The paper
     is NVLabs' (arXiv 2607.24027); kijai wrote the ComfyUI nodes.

---

## 6. Knobs: keep, remove, consolidate, change

| Knob | Evidence | Recommendation | Confidence | What settles it |
|---|---|---|---|---|
| `morton`, `morton_curve` | Off in all 131. Captures favour `3d` (`2026-09-17_sol_orderings.md`), but the blind panel split (`2026-09-18_sol_reorder_panel.md`). `2d_frame` and `hilbert` are already deprecated. NVLabs' H3 cells and LightX2V both say H3 takes no reordering (§9). Core's merged node never had it. | Fold into one off/3d switch and drop the deprecated curves. Keep the mechanism until the short-clip panel is scored; remove it if the panel does not favour it. | medium | score the panel |
| `tau_profile` | Never wired and never run. `dense_blocks` dominates it on the tail, and it is dead under top-k. No upstream varies tau per block (§9). | Remove, unless a per-block tau experiment is actually scheduled. | medium | the owner's call |
| `exact_kv_and_all_rows` | The 2026-09-04 probe moved text error and pixels. On 2026-09-10 the owner said "plan the switch", and pair C of `bench/sol_core_ab_arms.json` is still owed. NVLabs' Sana and the LongMedia pack both make **all** prefix rows dense (§9). | Keep. It is a pending default flip, not dead code, and upstream leans its way. | high | score pair C |
| `token_routing` + `token_aug_blocks` | Off everywhere, and no render has been judged with it on. The "all blocks" gate asks for more than the record supports, and the balance/token path has the §5.1 bug. | Make the list a sub-input of `custom`, the way `selection` works. Fix §5.1 first, re-grade, then relax the gate to balance alone if the fixed kernel agrees. | medium | a re-grade on the fixed kernel |
| `pooled_tail` | True in all 131. Setting it False without an SLA-distilled LoRA deletes Sol's own far-field term. | Move it under the top-k option; it is not a top-level knob. | high | none |
| top-k (SLA), `keep_percent` | None of the 131 use it. The Turbo-SLA LoRA arm retired unrendered on 2026-09-26. | Remove it together with `pooled_tail`, **if the owner confirms the SLA lane is closed**. Closed is not refuted, so that is the owner's call. | medium | the owner |
| `qk_balance` | Node default False, shipped True in 129 of 131, and probably inert behind the dense tail (§4.1). No independent upstream does anything like it (§9). | Render once with and once without it, same seed, shipped dense tail, and compare latents. If they are bit-identical, drop it from the recipe and keep the widget for experiments. If not, flip the node default to True. | medium | that render |
| `rotate` | In 2 graphs. It does most of balance's work (§4.4), has cost little since the warp port (`2026-09-17_sol_rotate_warp_port.json`), and its main benefit sits on blocks that are dense by default. | Consolidate with `qk_balance` into one quantizer combo (plain / balance / rotate / both). Pick the default after grading non-tail blocks (e.g. 40, 46, 47) on the installed build. | medium | that capture grade |
| `min_tokens` | 12288 everywhere. At the lengths we render it only catches the refiner calls. | Keep it for parity, but mark it advanced. | high | none |

**One constraint on any removal:** editor-saved graphs map widgets by
position, so removing or reordering a widget re-points the saved values. API
graphs are regenerated and unaffected (`sol_attn_h3.py` (lines 1678-1681 as of 2026-09-27; the file was restructured 2026-09-27),
`workflows/h3_config.py` (lines 966-968 as of 2026-09-27, before the redesign)).

---

## 7. What core has that we don't

| Core | Port? | Why |
|---|---|---|
| Re-installs itself on top every step (ON_PREPARE_STATE) | **yes** | Cheap, and it fixes §4.8's silent loss. |
| Uses the model's own `minimax_h3_layout` and block-patch args | **yes** | Ours monkeypatches `PackedLayout.__init__` process-wide and assigns `model._forward` and `rope_freqs` directly (`:507-533, :765-769`), the same irreversible-assignment shape that got the VSA node parked. Morton's hooks could move to `add_object_patch`. |
| Chunked producer through the block patch, with stats per conditioning branch and `pause_malloc_graph` | only if memory-bound arms matter | It never builds Q/K/V. It has no `qk_balance`, which is probably inert anyway (§4.1). Carried stats are not the same numbers as per-call stats (`2026-09-10_sol_impl_capture_grade.json`). `MiniMaxH3SolChunked` exists but needs a sage forward below it. No upstream besides core carries stats across steps (§9). |
| Accepts fp16; casts fp32 to bf16 | later | Once core PR #16508 lands. |
| `sla` and `vsa` in one node | no | Our VSA node is parked, and FastH3 uses core's. |
| `extra_tokens` as one global int (256) | no | Global routing is exactly what measured worse on block 49 without balance. |
| Per-step `ON_CLEANUP` reset; verbose off by default | no | No behavioural value. |

One thing core does worse: it never clears `block_index`, so refiner calls
see a stale index. That is the bug our hooks fixed. It is output-neutral as
long as `min_tokens` exceeds the refiner length.

---

## 8. Why we differ

Condensed from the first subagent's table, which gives the dates and
records. Core's choice came first only for `end_percent`, which we adopted
at 1.0 on 2026-09-11. Every other difference either predates core's node or
is an H3-specific lever added later.

| Difference | Reason | Kind |
|---|---|---|
| tau 1.0 vs 1.3 | Owner, 2026-08-20, from kijai's remark that 1.0 is max quality (`workflows/h3_config.py` (lines 213-253 as of 2026-09-27, before the redesign)). sglang, Sana and LongMedia's upper bound also use 1.0 (§9), so the adopt-upstream rule does not decide it. | reasoned, not measured |
| token routing off vs 256 everywhere | `docs/research/2026-09-04_sol_token_aug_grade.md`: better on 4 captured blocks, worse on 49. | measured |
| dense tail 45/48/49 | Owner, 2026-09-25. Kitchen dense int8 beats Sol's routed error on block 49 (`2026-09-15_ck_int8_attention_block49.json`). | measured on a capture; the render is unscored |
| `qk_balance` on | Owner, 2026-09-15, together with the kitchen floor. Lower block-49 error, neutral on blocks 0 and 32. | measured on captures |
| direct entry vs chunked | An override receives Q/K/V after `qkv_proj` and RoPE, so the chunked saving is already spent (`sol_attn_h3.py` (lines 80-86 as of 2026-09-27; the file was restructured 2026-09-27)). | structural |
| falls back on a kernel exception | Inherited from the vendored node. | no recorded reason |
| installs once | none | no recorded reason |

---

## 9. Upstream: who else does it our way

From the upstream-survey subagent, which read every `coderef/` checkout.
lookingdude spot-checked three claims against the source:
- sglang's Sol defaults of `dense_steps` 10 and `dense_layers` "0,1"
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/sol_attn.py:79-80`);
- Sana Sol-H3's prefix sink with every prefix row dense;
- LightX2V's refusal of Morton for H3.

Four sister repos are the owner's own or copies of our code and are not
counted as independent: the sage fork, ComfyUI-H3-Quant, and the T8 pack's
vendored Sol node and FastH3 V2.

| Technique | Who does it | Relation to us and core |
|---|---|---|
| Hadamard rotation of q/k before low-bit attention | vllm-omni (sign·H/√d on q and k before FP8/MXFP dense and sparse attention, NPU, `coderef/vllm-omni/vllm_omni/diffusion/attention/backends/flash_attn.py:595-600`); vllm (int4 KV cache, the same form); sglang (plain Hadamard on its sparse *indexer*); Model-Optimizer (fake-quant, no signs); kitchen's dense `int8_attention` (the same matrix as ours). None in FlashInfer, NVLabs Sol, FastVideo, TurboDiffusion, LightX2V or diffusers. | The technique is established upstream. **Doing it inside a sparse Sol kernel's int8 quantizers is ours alone.** |
| Per-channel q·f / k/f balancing | Nobody independent. Model-Optimizer's SmoothQuant is Linear-only, and FastVideo's attention QAT rejects `smooth_q`. K mean-smoothing (sage, FlashInfer, TurboDiffusion, kitchen's `kmean`) is common, and is a different thing. | **Ours alone.** |
| Top-k count rounding | `round`: kitchen, and so ours and core. `ceil`: FastVideo, sglang VSA-H3. `floor`: TurboDiffusion, LightX2V and T8 SLA, Model-Optimizer VSA. | The kitchen side of the VSA mismatch in the parity record. |
| Forced ±1 diagonal | kitchen, kijai, NVLabs Sol (`triton_ref/fwd.py`), LongMedia. Not forced in FastVideo, sglang VSA, vllm-omni or Model-Optimizer; sglang SubBlock leaves it out on purpose, citing a measurement. | Sol-family practice; foreign to VSA's trainers. |
| Conditioning as exact sinks, with dense query rows | NVLabs Sana Sol-H3 `sink_mode="prefix"`: exact KV **and all prefix rows dense** (it cites a clip whose dialogue fell apart). LongMedia does the same under core's name. FastVideo, sglang and vllm-omni VSA: prefix keys kept and prefix queries dense. | Upstream makes **all** conditioning rows dense, which is our `exact_kv_and_all_rows`. Dense target-audio rows only, our shipped default, is ours and core's alone. |
| Per-layer dense exemptions | Everyone upstream exempts the **first** layers: sglang "0,1", Sana 2, LightX2V [0]. FastVideo takes a list, empty by default. | **Dense last blocks (45,48,49) is ours alone**, and we exempt no first layers. |
| Dense warm-up | Counted in **steps** upstream: sglang 10, Sana 1 or 10, LightX2V 6 or 1, FastVideo 0. A percent window appears only in T8's SLA. LongMedia interpolates tau over sigma (1.3 to 0.8). | Our 0.2 start percent and core's are the same idea in different units. |
| Tau in score sigmas | NVLabs Sol, sglang (1.0), Sana (1.0), LightX2V (1.5), LongMedia. | We match sglang and Sana at 1.0; core's 1.3 matches none of these directly. |
| Extra per-token routing | Kitchen's `token_aug` only (core passes 256). | Kitchen-only mechanism. |
| Pooled tail | Kitchen: per query-block centroid. NVLabs Sol and LongMedia's port: per query **row**, which is finer. VSA uses a gated coarse softmax; SLA a learned linear branch. | Ours and core's tail is coarser than the paper's reference. |
| Token reordering | Sana says H3 needs none; LightX2V refuses Morton for H3; Hilbert appears nowhere else. | Upstream is against it for H3. |
| Statistics carried across steps | Only kitchen's `sol_attn_chunked`, reached through core's node (and T8 FastH3 V2). Everyone else computes per call. | Core's choice is the outlier; ours is the norm. |

Three things in this table weigh on §6:
- **All rows dense.** Upstream's all-rows default strengthens the case for
  flipping `sink_conditioning` to `exact_kv_and_all_rows`.
- **First layers dense.** Upstream's first-layer exemptions (sglang "0,1")
  are a lever we don't use and have not measured.
- **The pooled tail.** The paper's reference computes it per row, while
  kitchen uses one centroid per 64-row query block. That is a kitchen-level
  approximation our node and core share.

---

## 9b. Under the sage chain instead of the kitchen chain

The owner asked whether any of this changes when `MiniMaxH3SageAttention`
("fp8++ rotated", the sage chain's mode) sits under Sol in place of core's
kitchen backend. The alternate chain is built with
`python workflows/build_workflows.py --chain sage`; the owner's position
since 2026-09-17 is that neither chain has won (`h3_config.DENSE_CHAINS`).

### How calls are routed

The sage node patches every block's `attn.forward` and also registers an
override. Sol wraps that forward in its compose gate
(`sol_attn_h3.py` (lines 1241-1291 as of 2026-09-27; the file was restructured 2026-09-27)).

- **The gate** checks `min_tokens` and the sigma window. It does **not**
  check `dense_blocks`.
- **Calls the gate takes** run the sage node's `sol_take_forward`: sage's
  projection, rope and memory handling, then `optimized_attention`, then
  Sol's override. That covers dense blocks inside the window.
- **Calls the gate declines** (outside the window, short calls) run sage's
  forward directly.
- **Inside the override**, a dense block goes to `previous`, which here is
  sage's override. So dense blocks also run on sage, reached through the
  override path rather than sage's own forward.

### What changes and what doesn't

| Part | Change |
|---|---|
| Sol's own calls (47 blocks, steps inside the window) | **None.** Sol quantizes in its own kernel, so sage's mode never reaches it. Sol's `qk_balance` and `rotate` mean exactly what they did. Sage's rotation does not make Sol's `rotate` redundant. |
| Dense calls (first 20% of steps, 45/48/49, refiner, masked) | They run on sage fp8++ rotated instead of kitchen `int8_attention`. Both rotate q and k with the same sign·H128 matrix. Sage takes V and PV in FP8 with fp32+fp16 accumulation and has `smooth_k` off. Kitchen takes V int8 per channel, accumulates PV in int32, and shifts K by an anchor key. |
| `MiniMaxH3SolChunked` | It works only in this chain: it needs a foreign forward patch under Sol (`sol_chunked_h3.py:36-50`). The memory-saving chunked Sol entry is a sage-chain option. |
| Node order | The sage node must sit before Sol. Placed after, its override lands on top and Sol loses the calls (§4.8). |

### Which kernel is more accurate on the dense calls

- **Against plain sage:** kitchen int8 had lower error than plain sage
  fp8++ at every captured cell, steps 4 and 15. For example, block 49 at
  step 4 was 0.0158 against 0.0430, with the bf16 floor at 0.0017
  (`2026-09-15_dense_kernels_by_step.json`).
- **Rotation's effect on sage:** rotation cut sage's block-49 error from
  0.0549 to 0.0239 at step 15, and moved the other captured blocks by a few
  percent (`2026-09-17_sage_qk_rotate_kernel.json`, `kernel qk_rotate`).
- **Head to head:** I found **no record that grades sage rotated against
  kitchen int8 on the same heads and cells**. The two records use different
  head sets, so their numbers don't compare. Which kernel is more accurate
  now is unmeasured.
- **Renders:** on the market seed, all three kitchen-dense arms lost the
  hand in the coin beat, and the sage and bf16 arms kept it
  (`2026-09-15_block49_community_chain.md`). That was one seed, with the
  mechanism unknown, and a second seed is owed.

### What fp8++ rotated makes redundant

- **Sage-side balance.** Under rotation the balance factor finds nothing to
  do. The same record's `rotated+qk_balance` cells equal `rotated` on every
  block but 49, and differ there in the sixth digit. So sage's "fp8++
  balanced" mode and a `MiniMaxH3ChannelBalance` node on top of the rotated
  chain are redundant. The chain spec already leaves both out
  (`workflows/h3_config.py`, `DENSE_CHAINS` comment;
  `2026-09-17_channel_balance_vs_sage_balanced_b49_s15.json`).
- **Nothing on the Sol side.** Sol's `rotate` and `qk_balance` act inside
  Sol's kernel, which sage never touches.
- **Possibly the dense tail.** `dense_blocks` 45/48/49 was justified by
  kitchen dense beating Sol's routed error on block 49. Whether sage rotated
  also beats Sol there is not measured on matched cells. Until it is, the
  dense tail's benefit under the sage chain is an assumption carried over
  from the kitchen chain.

## 10. Not verified here

- That `qk_balance`'s gate stays shut on blocks 1-44, 46 and 47 (§4.1).
  It does not: measured 2026-09-27, `2026-09-27_sol_redesign_test1.md`.
- How large §5.3 is.
- Whether #208 changes int8 numerics for H3's call (fastdude's after-run).
- Kijai's reason for leaving `morton` out of the merged node. It is recorded
  nowhere I read.
- The upstream survey's claims beyond the three spot-checked.
