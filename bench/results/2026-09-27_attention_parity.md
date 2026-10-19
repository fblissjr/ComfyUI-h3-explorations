# Sol-Attn and VSA: our nodes against core's, and core against FastVideo (2026-09-27)

Read-only code comparisons, prompted by the owner's questions. Two were
subagents of this session; the third was the lookingdude session (kitchen fork
against upstream). There are no measurements here. Citations are file:line
as of this date.
- core: ComfyUI's `comfy_extras/nodes_sparse_attention.py` (NSA) and
  `comfy/ldm/minimax/model.py`;
- kitchen: the installed `0.2.35+sol.8176242`;
- FastVideo: `coderef/FastVideo/.../video_sparse_attn_h3.py`.

## Kitchen: the fork against upstream (lookingdude)

- **Merge base and additions:** both sit at v0.2.35, which ComfyUI pins.
  `h3-build` adds three opt-in features (blk_cnt, qk_balance, rotate).
  Upstream's 13 newer commits touch no sol_attn file. They do change
  `int8_attention`, the dense fallback, and those numerics are not checked.
- **What core's node gets:** it passes none of the fork-only arguments, so it
  gets upstream semantics from our build.
- **Our Sol node needs the fork as shipped:** `qk_balance=True` in
  `h3_config`. On a stock wheel `_apply_patch` raises loudly.

## VSA: ours (`vsa_attention.py`) against core's `vsa` against FastVideo

**The same in all three:**
- 4x4x4 token cubes, in t-h-w order and row-major inside a cube (checked on
  four layouts with a CPU snippet);
- edge padding;
- 64-row tiles for text and audio;
- the gated coarse branch, `gate(x)` with no activation, then
  `out + coarse * gate`;
- `tail=False`;
- RoPE and scale.

**Kitchen (ours and core) against FastVideo:**
- **Selection:** int8 pooled scores against fp32. The kept count is
  `round(0.2n)` against FastVideo's `ceil`, plus a tie back-off that keeps
  extra blocks.
- **The ±1 diagonal is forced exact** (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_route.cu:162`); FastVideo
  forces none. In cube order that means the w-neighbour cubes.
- **Precision:** int8 Q/K/V against bf16. Kitchen fixes all three. No
  setting of core's node reaches them, and our kitchen fork could add a
  diagonal flag but not bf16.

**Core against ours:**
- Core runs `sol_attn_chunked` with the previous step's K-mean and V scale
  (×1.1 margin, so V can clip). Ours quantizes per call. Both are int8.
- Core's `start_percent` defaults to 0.2, which is off FastH3 V2's contract.
  `FASTH3_CONTRACT_VSA` sets 0. Both nodes default keep to 10, where V2
  needs 20.

**Bottom line:**
- Neither matches FastVideo. Ours is marginally closer (per-call
  statistics).
- A FastVideo-exact reference is possible only in our own node: torch
  selection with fp32 scores, `ceil` and no diagonal; bf16 FlexAttention with
  a 64-row BlockMask (torch 2.14 has it); and a torch coarse branch. That is
  a moderate rewrite, and a reference, not a fast path.
- A VSA mode on our Sol node is not worth it. Almost every Sol knob would be
  forced off-default, and the result would copy core's deviations.

**Safety:** stock core builds `to_gate_compress` since e308cc73 (#16072), so
our node's old refusal passed and its `_publish_layout` leak became
reachable. The node is parked explicitly (0.157.1). The durable fix is
deletion: core already publishes `minimax_h3_layout`, and
`extra["original_block"]` replaces the hand-mirrored block.

**The dense backend on FastH3's graph:** the comparison says the two
token-refiner calls go through `ModelAttentionBackend`. But switching it to
bf16 gave bit-identical latents (`2026-09-27_fasth3_bf16rows.md`), so on
this graph those calls do not reach it, or it does not change them. Not
resolved further.

## Sol-Attn: our `MiniMaxH3SolAttn` against core's `sol-attn`

**The same:**
- the sink formula (both derivations agreed cell for cell on 2026-09-10);
- the sigma window (start 0.2, end 1.0);
- min_tokens 12288;
- pooled tail on;
- native token order;
- scale;
- falling back to the previous override.

**Policy at our shipped settings against core's defaults:**
- tau 1.0 against 1.3;
- token routing off against `token_aug` 256 on every block;
- dense blocks "45,48,49" on kitchen int8 against Sol on all 50;
- qk_balance on against none (core cannot run it: `sol_attn_chunked` has no
  such argument).

**Structural, at any setting:** core calls `sol_attn_chunked` (in-tile RMSNorm
and RoPE, previous-step K/V statistics, no full Q/K/V). Ours calls `sol_attn`
on full bf16 Q/K/V with per-call statistics. No setting of either node gives
the same bytes. Our `MiniMaxH3SolChunked` reaches the chunked kernel, but
passes no `token_aug` and keys its statistics differently.

**Robustness:**
- Ours falls back to dense on a kernel exception; core fails the prompt.
- Ours refuses non-bf16 H3; core silently runs it dense.
- Core re-installs its override every step; ours installs once.

**Better or worse, from the records:**
- **At matched knobs** (2026-09-10, `2026-09-10_sol_impl_capture_grade.json`):
  both nodes were the same distance from exact attention.
- **Ours at the then-shipped policy against core's defaults:** ours was better
  on the whole-tensor mean because of block 49, and core was better on all
  four block-0 cells and on the per-row mean.
- **Token routing** (2026-09-04 and 09-15 records): it helps blocks 0-40 and
  hurts block 49 unless qk_balance is on.
- **qk_balance:** lowers block 49's error and is neutral on blocks 0 and 32.
- **Ours as shipped today has never been compared with core.** qk_balance
  on, the kitchen fallback and the dense tail all postdate that one
  comparison. The rendered A/B pairs from 2026-09-10 were never scored.

**The GPU parity test, for later:**
- `bench/grade_sol_impl_on_capture.py` already has the arms. Add:
  - `qk_balance=True` to ours_shipped;
  - a dense-tail arm (int8_attention on 45, 48, 49);
  - a "core at our knobs" arm.
- The 2026-09-10 courtroom capture (blocks 0/24/49) is on disk until
  2026-10-31 and regrades with no render.
- A new capture adds blocks 45 and 48: one 345-frame render, about 108 GB of
  captures.

## Prose the code contradicts, found by these comparisons

Being corrected in the same pass as this record; see `docs/state/decisions.md`.
- **`vsa_attention.py`:** its docstring and refusal texts ("cannot run", "core
  has no gate_compress", "unreachable"); "Nothing has been rendered"; "the
  only other implementation"; its tooltip's "0.90 sparsity".
- **`docs/research/vsa/2026-08-30_vsa_node.md`:** its "core is stock, no
  gate_compress" passages.
- **`bench/check_vsa_core_patch.py`:** its "applied from PR #15958 DRAFT"
  message. The support is now in stock core.
- **`docs/attention/sol_attn.md`:**
  - the dense-layers row, with its columns swapped;
  - "ship empty as of 2026-09-02";
  - qk_balance described as off, or as a declared deviation;
  - "differ by no more than the all-routed floor" and "ours sits closer to
    exact", both contradicted as worded by the 2026-09-10 record (also in
    `docs/state/next_steps.md`).
- **`sol_chunked_h3.py:22-27`:** the routing threshold uses the current
  `kcvar`; only the K int8 centering and the V scale are stale.
- **`sol_attn_h3.py` (lines 13-15 as of 2026-09-27; the file was restructured 2026-09-27):** "every local change is listed below", when
  there are more now.
- **`sol_attn_h3.py` (lines 1506-1507 as of 2026-09-27; the file was restructured 2026-09-27) and the `h3_config.py` min_tokens comment:**
  both name sage as the fallback, where the shipped graphs use the kitchen
  backend node.
- **`workflows/h3_config.py` lines 846-849 at `56ccfb83`:** `SOL_CUDA_DEFAULTS` is called "untouched", but
  it pins qk_balance True.
- **`docs/attention/sol_upstream.md`:** line citations drifted by one, and the V-scale
  margin is not mentioned.
