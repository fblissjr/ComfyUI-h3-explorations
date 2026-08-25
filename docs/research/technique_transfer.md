# What transfers from LLM and ViT serving to H3, and what does not

Written 2026-08-25. A translation table for the owner's home field: each
serving technique from the LLM and ViT world, the property of the model it
needs in order to work, what it becomes for a bidirectional video DiT with a
prefill-only encoder, and what this repo has measured about it. It exists so
the next "could we borrow X" is answered by a row rather than a session, and
so the two borrows that are still open are named beside what would count as
measuring them.

Evidence labels are inside the claims. Status column: **exists** means built
and measured here; **declined** means an owner decision with the record
cited; **possible** means no measurement exists here either way; **n/a**
means the property the technique needs is absent by construction.

## The two facts that sort everything

1. **There is no decode loop.** The encoder is one prefill: `comfy/text_encoders/minimax.py` runs Qwen3-VL once over the presentation and taps `intermediate_output`; no generation call, no KV cache carried between calls. The DiT denoises every row in parallel each step. SOURCE.
2. **Every row is updated every block.** `comfy/ldm/minimax/model.py` runs one joint attention over the whole packed sequence (text, reference rows, audio, video) with no mask, and adds the attention and MLP residual to every row, conditioning rows included. So a reference row's K and V at block 1 already depend on the video rows, which change every step. Nothing is prefix-shaped. SOURCE.

Where the time goes, MEASURED at one configuration
(`bench/results/2026-08-18_phase0_instrument.json`, 1024x768 with three
references): sampling was 87% of the render at 100% SM occupancy and pegged
power; load staging and everything else was the rest. The conditioning node
(encoder plus reference VAE encodes, node 5 in the generated graphs) was 4%
of the all-refs render and under 2% of the turbo renders in the 2026-08-18 to
2026-08-22 arm rows under `bench/results/`.

## Techniques that need a causal decode loop

| from the LLM world | needs | here | status |
|---|---|---|---|
| KV prefix cache (radix tree, prompt caching, HiCache tiers) | rows unaffected by later rows | the DiT violates it by fact 2. The encoder is causal and H3's presentation puts reference blocks before the prompt text, so "same references, new text" is a real shared prefix there; the ceiling is the conditioning node's share above, and only when references repeat with different text. ComfyUI's node cache already returns unchanged conditioning for free | n/a for the DiT; not worth building for the encoder |
| speculative decoding (DSpark, DFlash, MTP, EAGLE) | weight-read-bound decode steps where verifying k tokens costs one | nothing is generated token by token. The diffusion analog, a draft model proposing denoising steps for the target to verify, needs a step budget with slack and a trained draft; INFERENCE, unmeasured anywhere here | n/a |
| paged KV, continuous batching, session-aware eviction | many concurrent sequences | one user, one sequence | n/a |

Where these two do apply in the owner's workflow: the LLM that writes H3
prompts against the system prompts in
[`h3_references.md`](../h3_references.md)'s prompting references. DFlash
ships drafts for Qwen3 and Gemma 4 under SGLang, vLLM and llama.cpp
(README read 2026-08-25). Outside ComfyUI and outside this repo.

## Techniques that need many steps

| from the LLM world | here | status |
|---|---|---|
| early exit, layer skipping, mixture of depths | step caching (Cache-DiT, TeaCache): reuse a block's residual when it moved little since the last step | **declined**, owner decision 2026-08-20 ([`roadmap.md`](../roadmap.md)): the measured speedup was steps skipped on a 16-step schedule, and the 4-step arm has nothing to skip. sglang's audited H3 setting and its SSIM are in [`sglang_comparison.md`](sglang_comparison.md) |
| distillation to a smaller student | step distillation | **exists**: the Turbo LoRA 4-step arm. [`h3_ref2v_distillation.md`](../h3_ref2v_distillation.md) records that ref2v resists it |
| adaptive compute (stop when converged) | adaptive step count | same regime as step caching; nothing to trade at 4 steps |

## Techniques that shrink or sparsify attention

These act on the sampling share, and the ViT world has more to offer here
than the LLM world does.

| from the LLM/ViT world | here | status |
|---|---|---|
| learned sparse attention with block selection (DeepSeek DSA, NSA, MoBA) | Sol-Attn's block-sparse routing with an exact sink and diagonal; `MiniMaxH3SLARouter` is the hard top-k version the Turbo-SLA LoRA was distilled under, and [`SOLATTN.md`](../SOLATTN.md) records why Sol at `top-k` is a third attention rather than a cheaper spelling of the router | **exists**, the most measured thing in the repo |
| FlashAttention, fp8 attention | sage (quantised QK) and the SDPA kernels | **exists** |
| sliding window, local attention | video needs global attention across frames; Sol's sink and diagonal are the structured part | n/a |
| **token merging and pruning (ToMe, VidToMe, FastV)** | merge redundant video rows before attention and unmerge after, a quadratic win on the rows removed | **possible, unmeasured.** [`sol_upstream.md`](../sol_upstream.md) notes Sol-Engine ships token pruning for LTX and not for H3. Sol already exploits the same redundancy by sparsity, so the two may overlap rather than stack; that is the question. Measurement: grade merged against exact on captured activations first (the `bench/grade_sage_on_capture.py` pattern, controlled by construction), then a blind distribution under [`eval_comparison.md`](../eval_comparison.md) section 3, never a pair |
| dynamic resolution, patch schedules | canvas, frame count, reference latent rows; the encoder's own view got a knob on 2026-08-25 (`qwen_short_edge`) | **exists** as the sequence-length levers in [`h3_resolutions.md`](../h3_resolutions.md) and [`h3_input_impacts.md`](../h3_input_impacts.md) |
| token reordering for locality (Hilbert, Morton) | Morton reordering under Sol | **exists**; [`morton.md`](../morton.md) records that whether it reaches the output is unverified |

## Techniques that shrink weights

These act on the non-attention share (the phase-0 instrument found the GEMM
phases DRAM-saturated, so weight bytes are their cost) and on load staging
(the DiT does not fit resident, [`hardware.md`](../hardware.md)).

| from the LLM/ViT world | here | status |
|---|---|---|
| W4/W8 weight-only quantisation of the LLM (AWQ, GPTQ) | the encoder: this repo's W4A16 AWQ, [`h3_awq_encoder.md`](../h3_awq_encoder.md) | **exists**; v2 calibrating on native input as of 2026-08-25 |
| the same on the ViT/DiT | the DiT ships int8 ConvRot (rotation plus int8); `fp8_scaled` measured worse against the bf16 release ([`evidence.md`](../evidence.md)) | **exists at int8. The unexplored borrow is W4 DiT weights** in the SVDQuant/Nunchaku style, which those projects apply to Flux and Wan DiTs. Prize: residency on 24 GB and faster DRAM-bound GEMM phases. Risk: diffusion's sensitivity to weight error, and a rendered pair cannot judge it (the different-sample rule in `CLAUDE.md`). Measurement: the same capture-graded route, then a blind distribution |
| KV cache quantisation | no KV survives a step | n/a |
| activation quantisation (A8) | sage quantises inside attention; the GEMMs are not quantised at the activation | possible, unmeasured |
| NVFP4 | Blackwell only; a 4090 is sm_89 | n/a on this box |

## Systems-level

| from the LLM world | here | status |
|---|---|---|
| CUDA graphs, breakable graphs, torch.compile | ComfyUI core has a per-block graph path the H3 loop does not enable; sampling at video canvases is compute/power-bound so replay has nothing to remove ([`sglang_comparison.md`](sglang_comparison.md)); the single-frame small-canvas case is pending `bench/instrument_render_occupancy.py` | measured at one configuration |
| CFG batching, CFG caching | the distilled checkpoint has one denoise branch (sglang's `MINIMAX_H3_DEFAULT_BRANCHES`, SOURCE) | n/a |
| offload, prefetch, disk tiers | ComfyUI dynamic VRAM streams blocks through a prefetch queue; the calibration bridge uses a disk tier | **exists** |
| tensor and sequence parallelism | one card | n/a |

## The two open borrows, stated as experiments

Both attack a different share and both can be graded on captured activations
before anyone renders, which keeps them inside the controlled-comparison rule.

1. **Token merging on the video rows.** Question: does merging stack with
   Sol's sparsity or double-count the same redundancy? First measurement: at
   Sol's shipped settings, merge ratio against exact-attention error on the
   captured activations, with Sol off and Sol on. A merge ratio that costs
   nothing with Sol off and something with Sol on is the double-count answer.
2. **W4 weights for the DiT.** Question: does a rotation-plus-4-bit DiT hold
   the bf16 release's output closely enough to earn residency? First
   measurement: the fidelity grading that produced the int8-versus-fp8
   verdict, rerun with a W4 arm. If it does not beat fp8 there, stop.

Neither is scheduled. The v2 encoder work has the card through 2026-08-26.
