# A global temporal offset, and AdaLN rounding on the INT8 DiT

last updated: 2026-09-25

> **Provenance, 2026-09-25.** A research subagent (Claude) read core, sglang
> and vllm-omni, and measured on CPU, using real checkpoint tensors and the
> repo's existing post-RoPE captures. The session that commissioned it wrote
> this note. The scripts and full outputs are kept unshared under
> `internal/2026-09-25_offset_adaln/`; the figures below are from them. Where
> a figure has only that home, it is labelled.

## A. What a global temporal offset does

**The mechanism** (ComfyUI checkout `comfy/ldm/minimax/model.py`):
- `PackedLayout` gives every row a three-axis position `(t, h, w)`.
  - Text takes `t = 0..L-1`.
  - Media starts after the text and reference spans, on a 40 Hz clock.
- RoPE rotates every row, text included, with one frequency set shared by the
  three axes.
- Only part of each head is rotated, and only a third of that carries `t`. So
  a time shift touches a quarter of each head's dimensions.
- Attention sees only position differences: media-to-media logits are
  unchanged under a shift, to fp32 precision.

**What an offset can do:**
1. **Nothing for a seam.** The model has no absolute clock. Offsetting all
   media by O is the same as moving the text and still references O further
   into the past. Seams depend only on media-to-media distance, which does not
   change.
2. **It weakens the prompt, more each window.** On the repo's two t2va
   captures, applying the rotation to the media rows lowered the attention
   mass video queries put on text as O grew. It fell by up to about a fifth at
   O = 850, where the third window of vllm-omni's default plan starts, and by
   up to about a third at O = 1700, where the fifth starts.
   Audio queries moved the same way. This is a static perturbation of stored
   activations, not a re-sampled trajectory, and the per-capture table is in
   the internal outputs. Where the offset puts text-media distances past
   anything a 362-frame clip reaches, the model is outside the range it
   plausibly saw. That range is not established from training data.
3. **It conflicts with per-window prompts.** An offset tells the prompt
   "this window is later". That suits one prompt with global timestamps, and
   works against vllm-omni's per-window prompts, whose shot times are local.
   A hypothesis with a direction, unmeasured.
4. **Outside continuation**, an offset on references against the target
   could act as a knob for how strongly a reference binds. It is untestable
   today: every capture on disk is t2va.

**Correction to the continuation note:** LongMedia's offset is not uniform
(`coderef/ComfyUI-MiniMax-H3-LongMedia/temporal_positioning.py`,
`_SHIFTED_SEGMENT_KINDS`). It shifts `cond`, `ref_audio`, `audio` and `video`,
and leaves out two kinds:
- `cond_audio`: a keyframe's audio slides against its video. This is now
  confirmed at first hand.
- `ref_img`: core labels a reference video's visual rows `ref_img` but its
  soundtrack `ref_audio`. So an offset desyncs the audio and video inside a
  reference-video block. New.

vllm-omni shifts every media kind together.

**Verdict:** don't adopt it for continuation. It is worth a render only as a
deliberate "weaken the prompt" or "weaken the reference" knob. The cheap first
step for that is a ref2va capture measured the same way.

## B. AdaLN rounding and our INT8 convrot checkpoints

**What core does:**
- **At load**, the adaln projection is stored at the compute dtype (bf16).
  The pruned checkpoint's time table stays fp32.
- **At run time**, `_mod_scale_shift` applies `(1 + scale)` and `shift` as
  two in-place bf16 ops on a bf16 norm output. The final layer does the same
  affine in fp32, so core's blocks and final layer use different conventions.
- **The consumer** is every block linear, which is `int8_tensorwise` with
  convrot on our files. On CUDA the activation is Hadamard-rotated in groups,
  then quantized per token (per row) to int8, then fed to an int8 GEMM
  (`comfy/ops.py`, kitchen's rowwise convrot quantizer).

**Upstreams disagree, so the adopt rule does not fire:**
- vllm-omni keeps the norm and the affine in fp32 with one store (#7913).
- sglang, on curve checkpoints like ours, rounds the norm to bf16 and does
  the affine in fp32 with one store.
- diffusers eager and core double-round.

**Measured on CPU** with an emulation of the int8 path, matched to kitchen's
eager op:
- **Size.** Core's rounding adds about 5% to the per-GEMM error at the median
  over blocks. It is largest in blocks 0 and 1, where the modulation switches
  channels nearly off and bf16 cannot represent `1 + scale` near zero well.
- **Character.** The error is coherent: the same for every token of a
  modality at a given t. So it does not average out the way the int8
  activation noise does.
- **Next to what the checkpoint already carries,** it sits below the int8
  activation quantization error, and below the int8 weight rounding the trace
  doc measured (`docs/research/comfyui_h3_t2va_trace.md`).
- **Over one step through all 50 blocks,** the convention ranking flipped
  from seed to seed, so no end-to-end ordering is resolved.

**Verdict: real, small, nothing worth doing now.** A pack-level fix would
need a runtime patch of the modulation plus an fp32 reload of the adaln
projection. Unfused, the fp32 affine at this canvas and length needs a large
temporary per call. Sizing the effect on real activations needs a new capture
at the norm input; today's captures are post-RoPE q/k/v.

## One open question this raised, worth a probe

**Do LoRA'd layers stay on the int8 path at run time?**
- The runtime-LoRA arms (PDD, FlashGen, Turbo) put a LoRA on
  `MODELS["unet_fl2va"]`.
- Read from core: a `weight_function` patch disables the quantized path and
  gives a bf16 GEMM with no activation quantization. The dynamic low-VRAM
  path with requantization stays int8.
- Which applies on this box decides whether those arms run the same numerics
  as the base graphs. It needs a runtime probe, not a reading.
