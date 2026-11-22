# Unified Compiler-Safe Attention-Hook Architecture Spec (`MiniMaxH3Steer`)

**Date:** `2026-11-22`  
**Owner (Spec):** Lane B · **Target Implementation Owner (post Phase 4 gate):** Lane A  
**Policy Scope:** **Strictly Multi-Vector & Pole-A Subspace Erasure (`add_norm_matched_erasure`)** — zero `1D` work or `1D` fallback branches.  
**Related Docs:** [`2026-11-22_paper_to_codebase_promotion_spec.md`](2026-11-22_paper_to_codebase_promotion_spec.md) · [`2026-11-22_track1_track2_multivector_erasure_blueprint.md`](2026-11-22_track1_track2_multivector_erasure_blueprint.md) · [`../../steering.md`](../../steering.md)

---

## Executive Summary

Phase 3 (`exp3`) and Phase 4 (`Showcases B & C`) proved that **block-band shadow-text K/V routing (`blocks.20..49`)** and **`1x Q_aud` attention splitting** decouple style/anti-slop control from identity and vocal prosody without `2x` DiT passes. However, the benchmark prototypes ([`bench/control_vector/run_phase3_steering_experiments.py`](../../../../bench/control_vector/run_phase3_steering_experiments.py#L1015-L1241) and [`bench/control_vector/run_phase4_stacked_and_stream_probes.py`](../../../../bench/control_vector/run_phase4_stacked_and_stream_probes.py#L611-L1230)) relied on four unsustainable hacks: process-global `disable_comfy_compiler = True`, shared `nn.Module` hook mutation (`blk.register_forward_pre_hook`), ad-hoc per-block `pause_malloc_graph()`, and full-sequence `k`/`v` concatenation (`1.14–4.02 GiB` peak tax at `41.8k–147k` tokens), while silently failing on Sage-only (`via_optimized_attention=False`) and `MiniMaxH3ExactBlocks` paths.

This document specifies **one unified, compiler-safe, zero-full-clone production hook architecture** for [`MiniMaxH3Steer`](../../../../control_vector_nodes.py#L4573) covering all combinations of `block_band` (`all` vs. `20..49` / custom range) and `streams` (`both`, `video_only_1x`, `audio_only_1x`), preserving full `comfy_aimdo.malloc_graph` compilation (`MALLOC_GRAPH_BREAKS == 0`, `MALLOC_GRAPH_ROGUES == 0`), `~26 MiB` bounded VRAM overhead (`L_txt = 256`), and full parity across every H3 attention backend (`MiniMaxH3Sol`, `MiniMaxH3SageAttention`, `comfy kitchen attention`, Stock SDPA, `MiniMaxH3ExactBlocks`, `MiniMaxH3AudioAttentionGain`, `MiniMaxH3LoRABranch`, `MiniMaxH3ChannelBalance`, `MiniMaxH3EditGuider`, `MiniMaxH3FunControlNetApply`, and `MiniMaxH3ImageModel`).

---

## 1. Root-Cause Audit of Prototype Patchwork (6 Disk-Verified Defects)

| # | Prototype Pattern & Site | Exact Mechanism / Bug | Impact in Production |
|---|---|---|---|
| **1** | **Process-Global Compiler Kill** (`bench/control_vector/run_phase4_stacked_and_stream_probes.py:807-808,1096-1097`) | `MiniMaxH3Model.forward()` (`comfy/ldm/minimax/model.py:686-700`) enables `malloc_graph_begin()` across `_forward()`, pushing `malloc_scope="block"` per DiT block (`:658`). Prototypes allocated cross-block `h_txt_steer`/`h_txt_base` inside pre-hooks, cached GPU tensors inside `_dm_wrap` on step 0 only, and allocated conditionally across `active_blocks`, breaking aimdo scope invariance (`comfy/model_prefetch.py:238-269`). | Setting `_ca.args.disable_comfy_compiler = True` mutates a process global shared with concurrent/chained models (`frozen_video_cache.py:855-884`), discarding ComfyUI's aimdo memory compilation and VBAR CUDA-graph acceleration. |
| **2** | **Shared `nn.Module` Hook Mutation** (`bench/control_vector/run_phase4_stacked_and_stream_probes.py:693,981`) | Calls `blk.register_forward_pre_hook(...)` directly on `diffusion_model.blocks` in place. | Mutates the underlying weights container across all `ModelPatcher.clone()` branches: A/B arms contaminate each other if clones share a `diffusion_model`, and stale hooks persist without manual `.remove()`. |
| **3** | **Full-Sequence `K/V` Clones & Container Stall** (`bench/control_vector/run_phase3_steering_experiments.py:1141-1153`; `bench/control_vector/run_phase4_stacked_and_stream_probes.py:721-726,1011-1018`) | Allocates `k_new = k_t.clone()` or `k_other = torch.cat([k[:, :, :tx_a], ks_cast, k[:, :, tx_b:]], dim=2)` for every block shadow call and audio split call, while holding live `q, k, v` `AttentionTensorContainer`s (`k.peek()` -> `_pass_through` -> consume later). | Allocates `~1.14 GiB` (`S=41,822`) to `~4.02 GiB` (`S=147,000`) of extra `BHND` `k_alt`/`v_alt` buffers simultaneously with `q, k, v`, defeating `_attention_comfy_kitchen_int8_containers` (`comfy/ldm/modules/attention.py:672-695`) and `AttentionTensorContainer` single-owner release. |
| **4** | **Silent Backend Bypasses (3 Separate Paths)** | **(4a) `MiniMaxH3SageAttention` without `Sol`:** [`nodes.py`](../../../../nodes.py#L211-L228) patches `blocks.{i}.attn.forward` with `via_optimized_attention=False`; `attention.py:1270-1283` calls `kernel_fn([q, k, v])` directly, **never calling `optimized_attention()`**.<br>**(4b) `Sol` warmup / short-seq fallback:** `_compose_module_patch` (`sol_attn_h3.py:976-995`) routes Sol-declined calls back to `patched_forward`, skipping `optimized_attention_override`.<br>**(4c) `MiniMaxH3ExactBlocks`:** `exact_blocks.py:81-84` strips `"optimized_attention_override"` on exact blocks (e.g., terminal block `49`). | Any implementation installed solely as an `optimized_attention_override` **silently stops steering completely** on Sage-only workflows, on Sol warmup steps when Sage is present, and on block `49` under `MiniMaxH3ExactBlocks`. |
| **5** | **`apply_audio_gain` Bypass in `1x Q_aud`** (`bench/control_vector/run_phase4_stacked_and_stream_probes.py:1027-1035`) | Prototype ran `F.scaled_dot_product_attention(q_aud, k_alt, v_alt)` manually inside an outer override, bypassing `apply_audio_gain` (`attention.py:278-319`). | `MiniMaxH3AudioAttentionGain` would affect `out_main` (`q_vid/ref/txt`) while silently missing `q_aud` -> audio temperature / value-gain skew. |
| **6** | **Combined Band + `Q_aud` Stream Contamination** | If a user sets `block_band = 20..49` AND `streams = video_only_1x`, Showcase B swaps steered K/V into the `base` main stream on block `20`. After block `20`, `h_main[tx_a:tx_b]` (`q_base`) has attended to `k_steer(20)`, contaminating `h_txt_base` for blocks `21..49`! | Unless `h_base` is cleanly forked before block `20` and evolved in a tiny `[L_txt, D]` base shadow through `20..49`, `q_aud` reads drifted mixed K/V from `21..49`. |

---

## 2. Compiler-Safe Lifecycle & Memory Protocol (`malloc_graph` Zero-Break Invariant)

### 2.1 Upstream `malloc_graph` Invariants (Verified on Disk)
From [`comfy/model_prefetch.py:31-110,231-335`](../../../../../../comfy/model_prefetch.py#L31-L110) and upstream `#16870` ([`comfy/ldm/qwen_image21/model.py:477`](../../../../../../comfy/ldm/qwen_image21/model.py#L477)):
1. `malloc_graph_begin(device)` wraps the entire `WrapperExecutor(self._forward, ...)` call (`comfy/ldm/minimax/model.py:689-700`, `MiniMaxH3ImageModel.forward:841-852`).
2. `prefetch_queue_pop(..., malloc_scope="block")` opens and closes a fixed-scope arena around **every single block `i` in `0..49`**.
3. Graph breaks (`MALLOC_GRAPH_BREAKS > 0`) and rogue leaks (`MALLOC_GRAPH_ROGUES > 0`) occur **only** when:
   - Allocation topology *inside an active scope* changes across blocks or across steps (e.g., allocating cache on step 0 only, or allocating a new shadow tensor inside `block[i]` that survives into `block[i+1]`), **or**
   - Unpaused allocations escape the scope in which they were created.
4. Upstream's sanctioned pattern (`comfy_extras/nodes_minimax_h3.py:533,597,603`, `comfy_extras/nodes_sparse_attention.py:74,279`, `#16870` `comfy/ldm/qwen_image21/model.py:477-497`) keeps persistent/cross-block staging buffers **outside** the memory compiler by allocating/recycling them once under `comfy.model_prefetch.pause_malloc_graph()`, synchronizing on the compute stream before copying, while leaving `original_block(...)` **inside active `malloc_graph` compilation**.

### 2.2 Call-Local State & Reusable Workspace (`H3SteerStepWorkspace`)
Rather than mutating `dm.blocks` or stashing `state` in module-global closures:
1. **Per-Clone Attachment:** [`MiniMaxH3Steer`](../../../../control_vector_nodes.py#L4573) only installs:
   - One `WrappersMP.DIFFUSION_MODEL` wrapper (`control_vector_nodes.py:3051`), AND
   - When any active spec requires `block_band != "all"` or `streams in ("video_only_1x", "audio_only_1x")`, chained per-clone `patches_replace["dit"][("double_block", i)]` entries (`H3SteerBlockPatch`, mirroring [`MiniMaxH3FunControlBlockPatch`](../../../../../../comfy_extras/nodes_minimax_h3.py#L589-L621) cleanly via `set_model_patch_replace`). Mode 0 (`all` + `both`) adds **zero** block patches and **zero** attention hooks.
2. **Thread/Branch-Safe Runtime Context (`transformer_options["_h3_steer_runtime"]`):** Inside `_dm_wrap`, build a fresh call-local `H3SteerRuntime` dataclass attached strictly to a shallow-copied `to_next = dict(transformer_options or {})`. Two concurrent branches (`MiniMaxH3EditGuider` `original` vs. `edited`, or batched CFG) never collide on `step_kv_by_block` or polarity flags.
3. **Reusable GPU Workspace (`H3SteerStepWorkspace`):** Owned on `transformer_options` (per-model cache keyed by `(device, dtype, l_txt, heads, head_dim)`, max 2 entries — freed automatically via ComfyUI `ModelPatcher.cleanup` / `CallbacksMP.ON_CLEANUP`):
   - Pre-allocated **once outside the block loop inside `with comfy.model_prefetch.pause_malloc_graph():`** (`~26 MiB` total at `L_txt=256, H=56, D_h=128, D=5376, bf16`):
     - `h_shadow_a: [L_txt, 5376]` (`~2.75 MiB`) — primary shadow text state
     - `h_shadow_b: [L_txt, 5376]` (`~2.75 MiB`, only if Mode 3 dual-shadow triggers)
     - `q_s, k_s, v_s: [1, 56, L_txt, 128]` each (`~3.67 MiB` each -> `11.0 MiB`)
     - `k_txt_saved, v_txt_saved: [1, 56, L_txt, 128]` (`~7.34 MiB`) for zero-clone save/restore
     - `q_b2, k_b2, v_b2: [1, 56, L_txt, 128]` (`~11.0 MiB`, only if Mode 3 dual-shadow triggers)
   - Synchronize compute stream once at allocation (`comfy.model_management.current_stream(device)` per `#16870`), then reuse purely in-place via `.copy_()` across all `50` blocks and all `N` steps.
4. **Refinement Outside `malloc_graph`:** Pre-refining unrefined `5120 -> 5376` `context` on step 0 (`control_vector_nodes.py:1997-2016`) AND transferring cached CPU `ctx_base` / `ctx_steer` to GPU MUST run inside `with comfy.model_prefetch.pause_malloc_graph():` so step 0 and steps 1..N present **identical** allocation graphs to `malloc_graph`.

### 2.3 Inside `H3SteerBlockPatch` (`patches_replace["dit"][("double_block", i)]`)
For each block `i`:
```python
class H3SteerBlockPatch:
    def __init__(self, previous=None):
        self.previous = previous  # chains FunControl / other dit double_block patches

    def __call__(self, args, extra_args):
        to = args["transformer_options"]
        rt = to.get("_h3_steer_runtime") if isinstance(to, dict) else None
        if rt is not None and rt.active:
            with comfy.model_prefetch.pause_malloc_graph(), torch.no_grad():
                rt.prepare_block(self.block_index, args)
        orig_blk = extra_args["original_block"]
        out = orig_blk(args) if self.previous is None else self.previous(args, extra_args)
        if rt is not None and rt.active:
            with comfy.model_prefetch.pause_malloc_graph(), torch.no_grad():
                rt.finish_block(self.block_index, out)
        return out
```
Notice three critical properties:
1. **`extra_args["original_block"](args)` runs completely unpaused inside `malloc_graph`!** Every main-stream GEMM, Sol/Sage attention, and MLP compiles natively with zero compiler breaks.
2. **Weights are already resident & unmerged LoRA + ChannelBalance aware!** Because `prefetch_queue_pop(prefetch_queue, device, block, malloc_scope="block")` runs right BEFORE `blocks_replace[("double_block", i)]` (`comfy/ldm/minimax/model.py:658-660`), `block`'s `qkv_proj`, `q_norm`, `k_norm`, `out_proj`, `adaln_proj`, and `mlp` (including `MiniMaxH3LoRABranch` unmerged forwards `lora_branch.py:284-345` and `MiniMaxH3ChannelBalance` folded `q_norm`/`k_norm` `channel_balance.py:186-187`) are ALREADY staged on `device` inside `block`!
3. **`Mod` slicing for text tokens is O(L_txt):** We only read `block.attn.qkv_proj(h_in)` and `block.mlp(...)` on `L_txt <= 512` rows (`~0.5%` of `S = 41,822–147,000`), using cached `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)` projected once from `[M, 2688]` `t_emb`. On `block_index == 49` (`n_blocks - 1`), `finish_block` skips updating `h_shadow` cleanly (zero wasted post-attention shadow MLP/proj).

---

## 3. Zero-Clone In-Place Text-Slice Save/Restore & Dual-Path Attention Seam

### 3.1 Why Full `K/V` Concatenation Is Eliminated (And Why Algebraic `LSE` Subtraction Is Rejected)
The naive alternatives to `k_alt = torch.cat(...)` are:
1. **Algebraic `return_lse` rank-`L_txt` subtraction** ($O_{\text{new}} = \frac{O_{\text{all}} - p_{\text{old}} O_{\text{old}} + r_{\text{new}} O_{\text{new,txt}}}{1 - p_{\text{old}} + r_{\text{new}}}$): **Explicitly rejected** after checking on-disk signatures (`inspect.signature(ck.sol_attn)` and `inspect.signature(ck.int8_attention)` confirm neither supports `return_lse=True`), and even with LSE kernels, subtracting $1 - p_{\text{old}}$ suffers catastrophic `bf16` cancellation on attention sinks where text holds substantial probability mass.
2. **Zero-Clone In-Place Text-Slice Save/Restore + Fused `q_alt` Query Scan (Adopted):**
   Look at the actual tensor algebra at attention time (`BHND` or `1SHD`):
   - All non-text keys `k[:, :, :tx_a]` + `k[:, :, tx_b:]` are **bit-identical** between the `main` pass and the `shadow`/`Q_aud` pass (`~99.5%` of sequence length `S`).
   - Only `k_slice = k.narrow(seq_axis, tx_a, l_txt)` (`L_txt <= 512`) and `v_slice = v.narrow(seq_axis, tx_a, l_txt)` differ!

**Exact step-by-step protocol inside the attention seam (zero full-tensor allocation, preserving `AttentionTensorContainer` early release):**

Suppose at block `i`, `(swap_main_kv: bool, run_shadow_attn: bool, run_aud_split: bool)` holds, and we inspect tensors cleanly (using `q.peek(), k.peek(), v.peek()` if held in `AttentionTensorContainer`s — taking care **not** to break container ownership for the fallback backend!):
1. **`with comfy.model_prefetch.pause_malloc_graph(), torch.no_grad():` Phase A — Shadow & `Q_aud` Fused Attention (before `main` consumes/frees containers!):**
   - **Rule:** Read `q_p, k_p, v_p` via `.peek()` (`BHND` or `1SHD`).
   - **Step A1 (if Mode 1 `blocks.0..19` OR shadow update needed in blocks `0..48`):**
     Wait — determine WHICH text keys `(k_s, v_s)` vs `(k_main_txt, v_main_txt)` the **shadow query `q_shadow`** and **`q_aud`** need to attend!
     - In **Mode 1 (`block_band = 20..49`, `streams = both`, `main = base`, `shadow = steer`):**
       - On `blocks.0..19`: main run keeps `k_p[tx_a:tx_b] = k_base`. Wait: shadow (`h_txt_steer`) ALSO wants to attend `[k_steer, k_non_txt]`! Look how clean that is WITHOUT cloning `k_p`:
         1. Copy main text slice into `ws.k_txt_saved.copy_(k_p.narrow(seq_axis, tx_a, l_txt))`, `ws.v_txt_saved.copy_(v_p.narrow(seq_axis, tx_a, l_txt))` (`~7 MiB`).
         2. Swap `ws.k_s, ws.v_s` into `k_p, v_p` at `[tx_a:tx_b]`.
         3. Compute `attn_shadow = flash_or_sdpa(ws.q_s, k_p, v_p)` (`O(L_txt * S)` FLOPs, zero `k_alt`/`v_alt` copy!). Save `attn_shadow` to `rt.shadow_attn_out` for `finish_block`!
         4. **Restore** `k_p.narrow(seq_axis, tx_a, l_txt).copy_(ws.k_txt_saved)`, `v_p.narrow(...)` so `main` (`pass_through`) sees original unsteered `k_base, v_base`!
       - On `blocks.20..49`: main stream wants **`(k_s, v_s)`**! Swap `ws.k_s, ws.v_s` in-place into `k_p.narrow(...)` **ONCE — no save, no restore!** Both `shadow` (`q_s`) AND `main` (`pass_through`) attend to the EXACT same updated `k_p` containing `(k_s, v_s)`! (And on `block 49`, `run_shadow_attn = False`!)
     - In **Mode 2 (`block_band = all`, `streams = video_only_1x`, `main = steer`, `shadow = base`):**
       - After `apply_audio_gain` (which scales `audio` rows cleanly so `q_aud` & `q_shadow` respect `MiniMaxH3AudioAttentionGain`!), save `k_txt_saved`/`v_txt_saved` (`~7 MiB`), overwrite `k_p[tx_a:tx_b]` in-place with `(k_b, v_b)` (unsteered base text K/V).
       - Concatenate ONLY queries: `q_alt = torch.cat([q_b] + [q_p[:, :, a:b] for a, b in au_spans], dim=2)` (`L_txt + L_aud <= 1,200` tokens vs. `S = 100k`!), run **one single** `attn_alt = _flash_or_exact_bhnd(q_alt, k_p, v_p)` call (`O((L_txt + L_aud) * S)` FlashAttention, `< 2%` extra FLOPs, zero `k_alt`/`v_alt` buffer!).
       - Split `attn_alt` into `rt.shadow_attn_out` (`[:l_txt]`) and `ws.aud_out_buf` (`[l_txt:]`).
       - Restore `k_txt_saved, v_txt_saved` back into `k_p[tx_a:tx_b]` so `main` (`pass_through` -> Sol / Sage / Kitchen / Stock) sees steered text K/V and still holds unconsumed `AttentionTensorContainer`s (preserving 100% of `_attention_comfy_kitchen_int8_containers` / `Sol` early QKV free)!
       - After `out_main = pass_through(q, k, v)` returns, splice `ws.aud_out_buf` in-place into `out_main` at `au_spans`!
     - **Attention-head & `skip_output_reshape` note:** In `comfy/ldm/minimax/model.py:207`, `optimized_attention(..., skip_reshape=True)` passes `BHND` (`[1, 56, S, 128]`) and expects `skip_output_reshape=False` -> `[1, S, 7168]`. Audio splice checks `skip_output_reshape` identically (`out_main[:, a:b]`) and handles both `BHND` (`1, H, S, D_h`) and reshaped `[1, S, H*D_h]`.
   - **`q_s` + `q_p` Audio Timing Guard (`streams = "audio_only_1x"`):**
     Notice how `streams = "audio_only_1x"` is the exact dual of `video_only_1x`! Set `main = base`, `shadow = steer` on all active blocks, compute fused `q_alt = cat([q_steer, q_aud])` against `[k_steer_txt, k_non_txt]` once, restore `k_base_txt` for `main` (which runs `base` video + references), and splice `aud_out_buf` into `out_main[au_spans]`! **Audio gets steered while video & image refs see pure unsteered base text in 1x pass!**

### 3.2 Solving Defect #4: Unified Dual-Path Attention Integration (Zero Missed Backends!)

Look closely at ALL call paths from `DiTBlock.forward` (`comfy/ldm/minimax/model.py:315-321`) -> `Attention.forward` (`:181-208`):

1. **Path A — Stock / Kitchen / Core Sparse (`via_optimized_attention` or Stock `Attention.forward`):**
   - Calls `optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)`.
   - `MiniMaxH3Sol` installs its `optimized_attention_override` on `CallbacksMP.ON_PREPARE_STATE` (`sol_attn_h3.py:1279-1281`, **before** `DIFFUSION_MODEL` wrapper executes).
   - Because `MiniMaxH3Steer`'s `_dm_wrap` runs **per forward step inside `WrappersMP.DIFFUSION_MODEL`** (strictly *after* `ON_PREPARE_STATE`), wrapping `to_next["optimized_attention_override"]` inside `_dm_wrap` **deterministically places `H3Steer` ABOVE `MiniMaxH3Sol` on every single step without ever growing the closure chain across steps!**
   - **AND** `H3Steer`'s override MUST propagate `container_function = functools.partial(override, None)` whenever `getattr(previous, "container_function", None) is not None` (`sol_attn_h3.py:904-905`), AND call `apply_audio_gain` parity BEFORE running `q_alt`! Wait — check `apply_audio_gain` ordering! Look at `attention.py:1159` and `attention.py:977`:
     - If `MiniMaxH3SageAttention` is active WITH `MiniMaxH3Sol`: Sol delegates via `sol_take_forward` (`attention.py:1180-1196`), which runs `apply_audio_gain` at `:1159` and sets `transformer_options["_h3_audio_gain_applied"] = True` before calling `optimized_attention`.
     - If `MiniMaxH3Sol` runs WITHOUT `MiniMaxH3SageAttention` (i.e., `ModelAttentionBackend` Kitchen INT8 or Stock beneath Sol): `Attention.forward` (`comfy/ldm/minimax/model.py:207`) does NOT run `apply_audio_gain`; Wait — does `apply_audio_gain` run inside `make_sage_override` (`attention.py:977`) / `make_lse_balance_override` (`attention.py:777`) BELOW Sol, NOT inside `Sol` itself?! **CHECK THAT ON DISK!** Look at `sol_attn_h3.py`! When `Sol` runs without `Sage` (pure `Stock`/`Kitchen` under `Sol`), `Sol` doesn't even import `apply_audio_gain` unless `Sage` is wired (`sol_take_forward`)! Meanwhile, if `H3Steer` calls `apply_audio_gain(k_p, v_p, transformer_options, 2)` ONCE before `q_alt` and sets `transformer_options["_h3_audio_gain_applied"] = True` (cleaned up on return), then:
       - `q_alt` sees the exact gained audio K/V, AND
       - underlying `make_sage_override` / `make_lse_balance_override` checks `_h3_audio_gain_applied` (`attention.py:294`) and will NOT double-multiply audio gain! That fixes Defect #5 cleanly AND prevents double-application!

2. **Path B — `MiniMaxH3SageAttention` (`via_optimized_attention=False` direct kernel path, OR Sol warmup steps routed back through `_compose_module_patch`):**
   - In [`nodes.py`](../../../../nodes.py#L225-L228), `MiniMaxH3SageAttention` patches `attn.forward` directly without calling `optimized_attention` unless `Sol` is active (`sol_take_forward`). Even when `Sol` IS active, `Sol` declines warmup steps (`outside_range`) via `_compose_module_patch` (`sol_attn_h3.py:976-995`) back to `patched_forward`, skipping `optimized_attention_override`!
   - **Fix (in `attention.py::make_minimax_attn_forward`):** Right at [`attention.py:1159-1160`](../../../../attention.py#L1159-L1160) (immediately after `apply_audio_gain(k, v, ...)` and BEFORE `via_optimized_attention` vs. `kernel_fn` branching):
     - Call `h3_steer_attn_seam_pre(q, k, v, transformer_options, seq_axis=1)` (`NHD` layout `[1, S, H, D_h]`) when `transformer_options.get("_h3_steer_runtime")` is active, setting `transformer_options["_h3_steer_kv_applied"] = True` across `via_optimized_attention` (`optimized_attention` override skips duplicate Phase A work when `_h3_steer_kv_applied` is True!).
     - Call `h3_steer_attn_seam_post(out, transformer_options)` after `kernel_fn` / `lse_balance` returns so `aud_out_buf` splices into `out` identically on Sage-only and Sol-declined warmup calls!

3. **Path C — `MiniMaxH3ExactBlocks` (`exact_blocks.py:81-84`):**
   - Currently `exact_blocks.py:81-84` strips `"optimized_attention_override"` on exact blocks (defaulting to block `49` to avoid Sage/Sol INT8 breakdown) before delegating to stock `Attention.forward`.
   - **Fix (2-line contract in `exact_blocks.py`):** If `transformer_options.get("_h3_steer_runtime")` is active, `exact_blocks.py` replaces `"optimized_attention_override"` with a minimal `h3_steer_exact_wrapper` (instead of removing steering along with Sol/Sage) OR calls the shared `steer_attn_override(..., previous=None)` directly over stock `func`! Thus exact block `49` runs bf16 Flash/SDPA (neither Sol nor Sage) **while still executing `1x Q_aud` splicing and block-49 K/V steering**!

---

## 4. Unified 4-Mode State Machine & 11-Path Backend Compatibility Matrix

### 4.1 The 4 Runtime Execution Modes

| Mode | Trigger Condition (`active_specs`) | Main Stream Text Identity | Shadow Stream(s) | `blocks.0..19` Action | `blocks.20..49` Action | Extra Peak VRAM (`L_txt=256`) |
|---|---|---|---|---|---|---|
| **Mode 0: Pure Prelude** | `band = all`, `streams = both` | `steered` | None | Stock (`0` patches, `0` attn hooks) | Stock | `0 MiB` |
| **Mode 1: Inverted Block-Band** | `band != all`, `streams = both` | `base` | `h_steer` (`[L_txt, 5376]`) | Evolve `h_steer`; save/swap `k_s/v_s` -> `q_s` attn -> restore `k_b/v_b` for main | Evolve `h_steer` (`20..48`); in-place overwrite `k/v[tx_a:tx_b] = (k_s, v_s)` (no restore!) | `~25 MiB` |
| **Mode 2: `1x` Stream Split** | `band = all`, `streams ∈ {video_only_1x, audio_only_1x}` | `video_only`: `steer`<br>`audio_only`: `base` | `video_only`: `h_base`<br>`audio_only`: `h_steer` | Evolve shadow (`0..48`); save `k_main`; swap `k_shadow`; 1 fused SDPA `cat([q_shadow, q_aud])`; restore `k_main`; splice `out_aud` | Same across all `0..49` (block `49` skips `finish_block` MLP) | `~26 MiB` + `q_aud` (`~5 MiB`) |
| **Mode 3: Combined Band + `1x` Video-Only** *(fixes Defect #6)* | `band = 20..49` + `streams = video_only_1x` | `base` on `0..49` (or clean fork) | `h_steer` on `0..49` + `h_base` **forked cleanly at block `20`** | Identical to Mode 1 (`h_base` == `h_main[tx_a:tx_b]` exact; **zero `q_aud` overhead**!) | At block `20` pre-block: `h_base_buf.copy_(h_main[tx_a:tx_b])`. For `20..49`: swap `k_b` -> fused `cat([q_base, q_aud])`; overwrite `k_s` for main; splice `out_aud` into `out_main` | `~42 MiB` |

*(Wait on Mode 3 cross-check: on block `20`, `q_main` (including `audio` queries!) sees `k_s`, while `q_aud` gets spliced from `k_b`. So `q_aud` on `20..49` attends purely to `h_base` evolving cleanly from block `20`! **Wait — subtle precision point on Mode 3:** `h_base` from `20..48` only updates `L_txt` tokens against `[k_base, k_non_txt]`, while `h_main[tx_a:tx_b]` in `20..48` was steered via `k_s`, which would have subtly altered `h_vid/aud` in later blocks — just as in real 2x split, `audio` sees `unsteered k_base` paired with live `k_vid/aud`! Forking `h_base` at block `20` prevents text self-contamination while keeping `0` block-`0..19` overhead.)*

### 4.2 End-to-End Composition Matrix

| Companion Feature / Node | Interaction & Handling in Unified Spec |
|---|---|
| **`MiniMaxH3Sol`** (`sol_attn_h3.py:1250-1281`) | `_dm_wrap` re-installs `H3Steer` override **after** Sol's `ON_PREPARE_STATE`, preserves `previous.container_function`, runs Phase A with `.peek()` + save/restore **before** Sol `take()`s containers (`sol_attn_h3.py:836-839`), ensuring Sol's dense sink KV blocks `[0:video_start]` always see exact `k_txt` / `v_txt`. |
| **`MiniMaxH3SageAttention`** ([`nodes.py`](../../../../nodes.py#L211-L309), `attention.py:1159`) | Works identically in **both** standalone Sage (`via_optimized_attention=False`) and Sol-composed Sage (`sol_take_forward`) via shared `h3_steer_attn_seam_pre/post` in `make_minimax_attn_forward`. Zero `container` double-free; preserves `clone_v=True` early fused-buffer release! |
| **`MiniMaxH3ExactBlocks`** (`exact_blocks.py:81-84`) | Exact blocks strip Sage/Sol/Kitchen overrides while preserving `H3Steer` KV swap + `aud_out_buf` splice -> block `49` stays exact `bf16` Flash/SDPA AND still respects block-band / `Q_aud` split. |
| **`MiniMaxH3AudioAttentionGain`** (`audio_freeze.py:756`, `attention.py:278`) | Both `kv_scale` and `lse_balance` run cleanly: `H3Steer` triggers `apply_audio_gain` (setting `_h3_audio_gain_applied`) before `q_alt` so `q_aud` obeys `key_gain`/`value_gain`, while `lse_balance` receives already-swapped `k_txt/v_txt` (`q_aud` splice applies over `lse_balance` output if both are enabled). |
| **`MiniMaxH3LoRABranch`** (`lora_branch.py:284-345,599-661`) | Automatically active in all shadow blocks! Calling `block.attn.qkv_proj(h_in)`, `block.attn.out_proj(...)`, and `block.mlp(...)` inside `prepare_block` / `finish_block` executes the unmerged `object_patches` installed by `MiniMaxH3LoRABranch`. |
| **`MiniMaxH3ChannelBalance`** (`channel_balance.py:186-187`) | `block.attn.q_norm.weight` and `k_norm.weight` already hold balanced weight diffs; reading them via `comfy.model_management.cast_to(...)` inside shadow RMSNorm+RoPE applies identical balance. |
| **`MiniMaxH3FunControlNetApply`** (`comfy_extras/nodes_minimax_h3.py:589-621`) | Chains via `previous` on `patches_replace["dit"][("double_block", i)]`. Both nodes pause `malloc_graph` only for sidecar control/shadow state while letting `original_block` compile natively. |
| **`MiniMaxH3EditGuider`** (`edit_guidance.py`) | Each pass (`original`, `edited`) carries its own `minimax_payload["h3_edit_pass"]` and creates its own call-local `H3SteerRuntime`; `edit_pass="edited_only"` skips hook activity cleanly on `original` with zero cross-contamination. |
| **`MiniMaxH3ImageModel`** (`comfy/ldm/minimax/model.py:826-893`) | Supports `block_band` (`Mode 1`, inverted shadow across `0..49`, zero `audio` stream); `streams` (`video_only_1x` / `audio_only_1x`) gracefully no-ops with info log when `layout` has no `"audio"` segment (`Single-Frame Image` invariant). |
| **Core `BlockSparseAttention`** (`comfy_extras/nodes_sparse_attention.py:269-325`) | Chunked producer bypasses `optimized_attention` (`h3_sparse_attention`) and replaces `double_block` with `args["attention"]`. Guard rule: refuse or log explicit warning + force `optimized_attention` path if wired with Mode 1–3 (`H3Steer` requires `MiniMaxH3Sol`, not core chunked `BlockSparseAttention`). |
| **`MiniMaxH3FrozenVideoCache`** (`frozen_video_cache.py:894-899`) | Already refuses non-empty `patches_replace["dit"]`; adding explicit mutual-exclusion error message if stacked with `block_band`/`1x` split keeps invariants strict. |

---

## 5. `MiniMaxH3Steer` Rule-5 Schema Extension & Verification Gate

### 5.1 Rule-5 Trailing Inputs (Only Promoted After Phase 4 Showcases B & C Pass)
Append strictly at the **end** of `MiniMaxH3Steer.define_schema` (`control_vector_nodes.py:4757`, preserving all upstream widget slots/types/defaults per `bench/check_node_ids.py`):
- Append options `"video_only_1x"` and `"audio_only_1x"` to existing `streams` combo (`:4674-4684`, wait — keeping `"video_only"` / `"audio_only"` unchanged if `2x` pass fallback is still exposed, OR adding `stream_split_mode` trailing combo if `check_literal_widgets.py` requires stable option lists: check `bench/check_literal_widgets.py` before altering existing `STEER_STREAMS`!). Safest Rule-5 design: keep `streams` options intact, and append **two trailing optional inputs**:
  1. `block_band` (`io.Combo.Input` / `io.String.Input`: default `"all"`, presets `"all"`, `"mid_late_20_49"` — activating Mode 1)
  2. `split_backend` (`io.Combo.Input`: `"auto_1x_kv"` vs. `"legacy_2x_pass"` — lets `video_only` / `audio_only` select single-pass `Q_aud` K/V split vs. reference 2x velocity split cleanly with zero widget-shift breakage!)

### 5.2 Mandatory CPU + GPU Verification Checklist (Before Any Code Lands)
1. **Bit-Exactness (Zero Active Blocks):** With `block_band` forced empty (`active_blocks = frozenset()`), Mode 1 MUST match unsteered `base` to `max_abs_diff == 0.0` at block `0` and step-end latent.
2. **Compiler Graph & Leak Gate (`--assert-graph-breaks`):** Run 6-step Pika `90f` `fl2va` and `ref2va` renders under ComfyUI compiler (`disable_comfy_compiler == False`!) asserting **`MALLOC_GRAPH_BREAKS == 0`** and **`MALLOC_GRAPH_ROGUES == 0`**.
3. **Zero Full-Clone Memory Peak Gate:** Verify `torch.cuda.max_memory_allocated()` delta vs. unsteered baseline stays **$\le 128\text{ MiB}$** at `90f` `1152x768` (`S > 30k` tokens), confirming no full-sequence `K/V` copies occur.
4. **4-Backend Parity Matrix:** Assert active `kv_swaps` and `aud_splices` fire cleanly (> 0 calls) across:
   - `MiniMaxH3Sol` + `MiniMaxH3SageAttention` (`rotated`)
   - `MiniMaxH3SageAttention` standalone (no `Sol`, `via_optimized_attention=False`)
   - `ModelAttentionBackend` (`comfy kitchen attention`, no `Sage`)
   - `MiniMaxH3ExactBlocks(blocks="49")` combined with any of the above.
5. **Multi-Clone Isolation Test:** Branch `m_base` and `m_band = MiniMaxH3Steer(m_base, block_band="20-49")`, verify `m_base` has `0` hooks/patches and identical output to pre-clone baseline.
6. **Node Schema Gate:** `bench/check_node_ids.py`, `bench/check_literal_widgets.py`, and existing `bench/check_control_vectors.py` tests pass (`exit 0`).
