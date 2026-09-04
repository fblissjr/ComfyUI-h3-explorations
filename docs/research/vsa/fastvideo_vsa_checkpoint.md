# The FastVideo VSA-distilled H3 checkpoint

last updated: 2026-09-04 (section 6 only; every measurement below is from
2026-08-30 and the artifact has not changed)

`minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors`,
published at `huggingface.co/Kijai/MiniMax-H3-experimental` and reachable here
as `models/diffusion_models/` plus that name.

**This file owns what is inside that artifact and how far its weights sit from
the base it was built on.** Everything numeric in it was produced by
[`../../../bench/analyze_vsa_checkpoint.py`](../../../bench/analyze_vsa_checkpoint.py)
into
[`../../../bench/results/2026-08-30_fastvideo_vsa_checkpoint.json`](../../../bench/results/2026-08-30_fastvideo_vsa_checkpoint.json),
except the load behaviour in section 5, which was produced by executing this
install's own detection and module construction against the checkpoint header.

**It does not own what the checkpoint is worth.** Nothing here was rendered,
nothing was timed, and no attention output was compared. It also does not own
sparse attention itself: [`../../SOLATTN.md`](../../SOLATTN.md) is the authority
for the kernel this gate feeds, and this file asserts nothing against it.

## 1. The whole difference from the base is the gate, and nothing else

Measured against `minimax_h3_fl2va_pruned_int8_convrot.safetensors`, the file
that ships in the pruned int8 representation this checkpoint is also in:

| | |
|---|---|
| keys present only in the VSA file | `blocks.N.attn.to_gate_compress.{weight, weight_scale, comfy_quant}`, N in 0 to 49 |
| keys present only in the base | none |
| shared keys whose shape or dtype changed | none |
| `__metadata__` | absent (the base carries a `config` blob; this file carries nothing) |

The parameter count difference is `1,927,120,400`, and the gate weights, their
fp32 row scales and their quant descriptors account for exactly that, with no
remainder. So this is a **full checkpoint**, not a partial one and not a LoRA:
it is the base state dict with one extra linear per main block and nothing else
added, removed or reshaped. On disk it is 1.796 GiB larger than the base.

The absence of `__metadata__` matters more than it looks. The base's blob is
where a reader would go for the transformer config; here there is no
provenance, no training record, no step count and no schedule. **Everything
this file claims about itself lives in its filename.**

## 2. The gate

| | |
|---|---|
| modules carrying it | the 50 main blocks, all of them |
| modules not carrying it | the 2 `token_refiner` blocks, and `final_layer` |
| shape | `[7168, 5376]`, which is `Linear(hidden=5376, heads*head_dim=56*128, bias=False)` |
| bias | no `to_gate_compress.bias` key exists |
| quantization | `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`, byte-identical to the descriptor on the same block's `qkv_proj` |

So the gate is **not** held back in a higher precision. It is quantized exactly
like every other block linear, including the Hadamard rotation, and it pays the
same rounding cost as the weights it sits beside.

The shape matches what Comfy-Org/ComfyUI#15958 constructs. That PR's diff
(*read, not built*) adds `gate_compress=False` to the H3 `Attention`
constructor and, when set, `operations.Linear(hidden, inner, bias=False)`,
described in its own comment as a "per-token gate for the coarse attention
branch"; detection keys on `blocks.0.attn.to_gate_compress.weight`.

**The gate is trained, not a placeholder.** No row of a sampled gate is all
zero, and about one int8 entry in eighty is exactly zero, which is what a dense
trained matrix quantizes to rather than what an initialized-and-frozen one
would. Its scale is close to flat with depth: the mean fp32 row scale across
all 50 blocks spans a factor of 1.35 from smallest to largest, so there is no
depth structure of the kind this repo found in the PDD LoRA deltas
([`../h3_partition_distance.md`](../h3_partition_distance.md)).

**What the gate is for.** The installed kernel consumes it as
`out += gate * softmax(q_mean k_mean^T * scale) v_mean` per block, applied raw
with no activation (`comfy_kitchen.backends.eager.sol_attn`, functions
`coarse_output` and `add_coarse_`, *read in the installed package*). The
third-party integration in
`coderef/comfyui-minimax-h3-audio-T8/fast_h3_vsa_advanced.py` computes it the
same way: `attention.to_gate_compress(x_padded)` on the modulated block input,
viewed as `(1, rows, heads, head_dim)`, handed to `sol_attn` as `coarse_gate`
alongside 4x4x4 cube tiling, per-block `block_len` padding and `tail=False`.
Since the gate multiplies the coarse branch's output and that output is in the
same units as the sparse branch it is added to, **the magnitude of the gate's
output is itself the mixing coefficient** between the two branches.

The checkpoint supplies one half of that magnitude and not the other. The gate
weight's mean row norm runs 2.5e-3 to 2.9e-3 against 5.6 to 9.9 for the same
block's `qkv_proj`. What that becomes at runtime depends on the activation
scale, which is in no checkpoint and was not measured here.

## 3. It is pruned, and it is fl2va

**Pruned.** The file carries `adaln_t_table` and per-block
`adaln_proj.linear.weight` of shape `[96768, 8]`, and carries no
`time_embedder.proj_in` or `time_embedder.proj_out`. That is the pruned form:
adaln shipped over a small shared basis of the time-embedding curve instead of
a time embedder and full-width adaln linears. Against the **unpruned** fl2va
file the same comparison reports the four time-embedder keys and the 100 adaln
quant sidecars absent, and 102 adaln tensors changed shape. That is the
pruning and nothing else.

**fl2va, decisively.** Key sets cannot answer this; the partitions have
identical key sets, which is why a wrong-partition load is silent. The numbers
answer it without ambiguity. Over the 200 block norms:

| | vs fl2va (pruned) | vs ref2va (pruned) |
|---|---|---|
| bit-identical tensors | 185 of 200 | 0 of 200 |
| median relative distance | 0 | 4.0e-3 |
| worst tensor | 3 elements differ, 1.8e-6 relative | 3821 elements differ |

and over every unquantized tensor on the render path, the fl2va distance is
between 24x and 350x smaller:

| tensor | vs fl2va | vs ref2va |
|---|---|---|
| `condition_proj.weight` | 9.7e-5 | 2.4e-2 |
| `token_refiner.blocks.0.attn.qkv_proj.weight` | 1.8e-4 | 3.3e-2 |
| `final_layer.audio_out.weight` | 5.9e-4 | 7.9e-2 |
| `video_patch_proj.weight` | 9.0e-4 | 2.2e-2 |
| `final_layer.video_out.weight` | 1.2e-3 | 5.0e-2 |

The ref2va column reproduces the partition distances
[`../h3_partition_distance.md`](../h3_partition_distance.md) measured
independently, which is the corroboration that the comparison is doing what it
claims.

## 4. How far the weights moved

The short answer: **the trunk did not move by more than a precision round trip,
and the only surface that moved measurably is adaln, where the movement has the
signature of a refit rather than of training.** Getting there needs two traps
routed around first, because both of them produce a large and completely
meaningless number.

### 4.1 Why a diff of the int8 payload is not a weight distance

`int8_convrot` stores `round(W @ H^T / s)` with a per-row fp32 `s`, where `H` is
a fixed orthogonal Hadamard of order 256 baked in offline
([`../comfyui_h3_t2va_trace.md`](../comfyui_h3_t2va_trace.md) section 1.7). Two
things follow, and they point opposite ways.

The helpful one: `H` is the same in both files, because both carry the same
`convrot_groupsize`, and an orthogonal transform preserves Frobenius distance
exactly. So a relative distance computed on `q * s` in the rotated basis **is**
the relative distance in the weight basis.

The unhelpful one: **the two files do not use the same rule for `s`.** In the
base, every row of every sampled linear saturates at `|q| = 127`, which is plain
per-row absmax. In the VSA file, between 57% and 82% of rows do. A different
scale rule means an unchanged weight quantizes to different bytes, so the int8
payload differing proves nothing, and the dequantize-requantize fixed point that
would otherwise let bytes stand in for weights does not hold.

What is left is `q * s` compared against the rounding error it carries. For
round-to-nearest the per-element error is uniform on plus or minus `s/2`, so
each file's error energy is `ncols * sum_r s_r^2 / 12`, computable from the
scale vector alone. Two independent quantizations of an *identical* weight would
land near the root sum of squares of those two.

| block | linear | measured relative | predicted floor | measured / floor |
|---|---|---|---|---|
| 0 | `attn.qkv_proj` | 8.7e-3 | 1.25e-2 | 0.70 |
| 0 | `attn.out_proj` | 9.8e-3 | 1.70e-2 | 0.58 |
| 25 | `mlp.fc1` | 8.7e-3 | 1.25e-2 | 0.70 |
| 49 | `mlp.fc2` | 9.3e-3 | 1.35e-2 | 0.69 |

Twelve linears were sampled across blocks 0, 25 and 49; every one lands between
0.58 and 0.70. **The measured distance is below the floor, on every sample.**
That is what an unchanged weight looks like: the floor assumes the two roundings
are independent, and where both files quantize the same numbers they are
partially correlated, so landing under it is the expected result and landing
above it would have been the finding.

This is an upper bound, not a proof of zero. Any real movement is buried under
roughly a percent of rounding, and this comparison cannot see it. The
rounding-insensitive statistic agrees: the per-row L2 norm profile, which
barely notices quantization, differs by 1.6e-3 to 1.8e-3 across all twelve.

### 4.2 The unquantized trunk, where there is no error model to argue about

Every tensor in the section 3 table is stored unquantized in both files, so
those fl2va distances need no interpretation. Across the full set measured they
run from 9.3e-5 to 1.2e-3, with cosine never below 0.9999993, and
`token_refiner.final_norm.weight` and `rope.inv_freq` are bit-identical.

The split between them is by dtype, not by role, which is the tell. The bf16
tensors agree to about 1e-4; the fp32 tensors to about 1e-3. So the control:
round each fp32 base tensor through bf16 and back, and measure what that alone
costs.

| tensor | VSA vs base | a bf16 round trip of the base |
|---|---|---|
| `video_patch_proj.weight` | 9.0e-4 | 8.9e-4 |
| `audio_patch_proj.weight` | 8.7e-4 | 8.5e-4 |
| `final_layer.audio_out.weight` | 5.9e-4 | 3.7e-4 |
| `final_layer.video_out.weight` | 1.2e-3 | 6.1e-4 |

Two of the four sit on the floor and two sit within a factor of two of it. The
distances are the size of a precision round trip, and the tensors are not
exactly the round trip either, so the honest reading is **at or near the
precision floor, with no room for an update large enough to matter, and not a
proof that the number is exactly zero.**

### 4.3 The adaln surface, and the trap that makes it look retrained

Compared tensor by tensor, `adaln_t_table` and the per-block
`adaln_proj.linear.weight` look catastrophically different: relative distance
above 3 with *negative* cosine, against both partitions. **That comparison is
meaningless.** The pruned model stores one shared table of shape `[1025, 8]` and
a per-block `[out, 8]` linear, and evaluates `t -> lerp(table)[t] @ W^T + b`.
Any invertible 8x8 basis change applied to the table and undone in every block
leaves the model bit-for-bit identical in behaviour and arbitrarily different in
storage. The tables here *are* in different bases, which is itself the finding:
kijai did not start from Comfy-Org's pruned file, he re-derived the pruning.

Comparing the affine map instead, exactly, through Gram matrices:

| | vs fl2va | vs ref2va |
|---|---|---|
| median over 51 adaln modules | 1.0e-2 | 3.3e-2 |
| range | 5.1e-3 to 1.3e-2 | 9.9e-3 to 4.8e-2 |

So adaln is the one surface that moved by more than the precision floor of
section 4.2, by about an order of magnitude. Two candidate readings, and the
measurement separates them partway.

Storing the 8-column factor in fp16 is **not** the explanation. The
factor-to-product amplification, `||T|| ||W|| / ||T W^T||`, has a median of 2.4,
so fp16's element-wise precision cannot reach 1e-2 through it.

A different rank-8 fit of the same underlying curve **is** consistent with what
the structure shows. The two surfaces' factor column spaces agree to within 1e-4
on their first seven directions, to 0.99 on the eighth, and disagree sharply
only on the ninth and last. The base surface's energy beyond its top three
singular values is under 1e-4 of the total, and the largest singular value
agrees to within a percent on every module sampled. That is two low-rank
derivations agreeing about everything that carries energy and disagreeing about
a near-null direction, which is not the shape a trained update would take.

**What would settle it, and has not been done:** reconstruct the true adaln
surface from the unpruned fl2va checkpoint by running its `time_embedder` over
the same grid, and measure which of the two pruned surfaces is closer to it. If
they are equidistant the difference is fitting; if the VSA one sits further from
the unpruned surface in a structured way, adaln moved. That needs the time
embedder's forward reimplemented correctly, which is the reason it is not here
rather than a reason it cannot be done.

### 4.4 The control that makes the section above readable

None of the above would mean much without knowing what "unchanged" looks like
in this file format. Comfy-Org's own pruned and unpruned fl2va files supply it:
they differ by the adaln pruning and by nothing else, and of the 829 tensors
whose shape and dtype survive the pruning, **every single one is byte-identical**
over the sampled prefix. So bit equality is achievable through this lineage's
tooling, and the 185-of-200 bit-identical block norms in section 3 are the VSA
file inheriting fl2va bits rather than reproducing them.

## 5. What running it needs, and what this install does with it today

**The kernel half is present.** The installed `comfy_kitchen 0.2.31+sol.dae00a1`
exposes `sol_attn` with `topk_ratio`, `tail`, `block_len` and `coarse_gate`
(*executed*: the signature was introspected), which is the merged
comfy-kitchen#117 the model card points at.

**The core half is not.** ComfyUI here is on `master` at `8a33128f`, and
`gate_compress` appears nowhere in `comfy/` or `comfy_extras/`. So
Comfy-Org/ComfyUI#15958 is not in this install. It is still a draft PR.

What that produces, *executed* rather than reasoned: feeding the checkpoint's
header to this install's `detect_unet_config` returns a config **identical** to
the base's, with no gate field, and the model built from it has no
`to_gate_compress` attribute on any block's attention. The 150 gate keys have
no slot at all.

*Read rather than executed, for the last step:* `comfy/model_base.py` loads the
diffusion state dict with `strict=False` and logs the unexpected keys as a
warning. So the load succeeds, warns, and drops 1.796 GiB of weights on the
floor.

**The operative consequence: on this install today, loading this checkpoint
gives you the fl2va base model running dense.** It does not fail, it does not
warn in a way anyone reads as an error, and there is no visual signature. If
you want to compare it against anything, that is the trap to avoid.

**Installing the PR is necessary and not sufficient.** Its own comment says the
gate is unused by the dense forward and consumed by sparse attention patches.
With the PR, the gate modules exist and the weights load; the dense forward
still never calls them, so the render is unchanged and the 1.796 GiB is now
resident instead of discarded. Something has to compute
`to_gate_compress(x)` and pass it to `sol_attn` as `coarse_gate`, along with the
cube tiling and `block_len` the kernel needs. Nothing in this pack does that,
and no shipped graph wires anything that does. The only implementation of that
step available here is the third-party
`coderef/comfyui-minimax-h3-audio-T8/fast_h3_vsa_advanced.py`, which is a
source to read, not something this repo has run or graded.

## 6. "4step"

**Nothing in the artifact supports it.** There is no `__metadata__`, no sigma
schedule, no step count, no scheduler config, no LoRA alpha, and no extra output
heads. The file carries exactly one `final_layer.video_out` and one
`final_layer.audio_out`, so whatever "4step" means here, it is **not** PDD,
which reaches its step count by replicating output heads over a 32-point grid
([`../../h3_pdd.md`](../../h3_pdd.md)).

**The model card does not say either.** Its only text about this file is that it
is "currently for testing with these PRs", linking Comfy-Org/ComfyUI#15958 and
comfy-kitchen#117. It names no sampler, no step count, no CFG and no shift.

So the claim rests on the filename alone, and the filename is the least reliable
carrier in this repo's experience. Two readings are available and this document
picks neither: that VSA's sparsity is what makes four steps viable, or that
"1300step" is the distillation length and "4step" the target the distillation
aimed at. **Do not write a 4-step recipe for this checkpoint from anything in
the artifact.** The cheapest thing that would settle it is asking, not measuring.

**Added 2026-09-04: the source is now public, and it answers the question the
artifact could not.** FastVideo published the "FastH3 Preview v1" collection
on 2026-08-27 (`huggingface.co/FastVideo`, repos
`FastVideo-FastH3-4-step-Preview-v1-{VSA-DataFree, Dense-DataFree, LoRA,
VSA-Synthetic-Step1300, VSA-Synthetic-Step1900}`), one day before kijai's
int8 conversion above was uploaded. The VSA-DataFree card describes the
weights as a step-1300, data-free DMD2 four-step distillation trained with
VSA-H3 at 0.9 sparsity on 64-token tiles, t2va only: the second reading
above, and a distillation of the sampler rather than of the attention. Two
serving engines have since pinned the recipe as code, and those are the
pointers to use rather than anything retyped here:

- sglang, `coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::FastH3SamplingParams`
  (five sigma grid points, four DiT forwards, `t2va` only, other step counts
  and tasks refused) and `coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::FastH3PipelineConfig`;
  the cookbook section "FastH3: 4-step distilled preview" in
  `coderef/sglang/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx` says the
  points sit on the standard shift-12/shift-3 grid.
- vllm-omni, `coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/fasth3.py::FASTH3_BASE_SCHEDULE`
  (the five positions as a constant, with the per-modality shifts applied on
  top) and the module docstring, which documents the adapter format the
  `-LoRA` repo uses: rank-64 factors at scale exactly one plus full-rank
  `.diff`/`.diff_b` deltas and `.set_weight` gate tensors, which is why no
  LoRA loader can apply it and both engines fuse it at load.

What that does not change: the artifact here still carries none of it, the
"4step" in its name is still a filename, and nothing here has verified that
this repo's sampler reproduces those five points. Whether the schedule is
worth a rung is a decision for the roadmap, not this file.

## 7. What was not checked

- **Nothing was rendered and nothing was timed.** No claim here is about output
  quality, speed, or whether the gate improves anything.
- **The gate's runtime magnitude.** Section 2 has the weight half of the mixing
  coefficient and not the activation half, so "how much does the coarse branch
  contribute" is open. It needs an activation capture, which this repo can do
  (`bench/capture_h3_encoder_states.py` is the encoder-side one; the DiT side is
  `h3_capture.py`) and which was not done.
- **The quantized weights below the rounding floor.** Section 4.1 bounds the
  movement at about a percent and cannot see under it. Recovering the true
  distance needs a bf16 source for both sides, and there is no bf16 VSA
  checkpoint.
- **Whether adaln was trained or refit.** Section 4.3 names the experiment.
- **Only three of fifty blocks** were sampled for the quantized and gate
  per-tensor work, chosen front, middle and last. The byte census and every
  unquantized comparison cover all fifty.
- **The prefix sample in the byte census** reads the first MiB of each tensor.
  A tensor reported as matching matched over its prefix, which is not the same
  as identical; the tensors small enough to be covered in full are the norms and
  the biases.
- **PR 15958 was read, not built.** Its behaviour here is inferred from its diff
  plus the executed fact that this install lacks it. Nobody has run the
  checkpoint with the gate live on this box.
