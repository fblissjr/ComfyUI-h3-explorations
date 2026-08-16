# Roadmap: what we are trying to find out, and what would count as finding it

Last updated: 2026-08-16.

## What this is, and who it is for

A working plan for the H3 speed-and-quality investigation, written so a fresh
session can pick it up without re-deriving the constraints or re-litigating
decisions that are already made.

**The context matters and shapes everything below.** This is one person
tinkering on a single RTX 4090 with 24 GB, finding out what is possible with
MiniMax H3. It is not a research programme and it is not production R&D. There
is no team, no cluster, no deadline, and no obligation to be exhaustive.

The practical consequence: **card time is the scarce resource and rigour is
cheap.** A measurement that takes a minute of Python and settles something is
worth more than an afternoon of renders that produces a number nobody can
defend. Most of the good results in this repo came from that trade -- the
canvas cost table, the Morton geometry, the ref LoRA grading, the MMA rates
all needed zero or one render.

Two companion documents, and the split matters:
`docs/open_experiments.md` is the list of things **not measured** and why.
`docs/evidence.md` is the ledger of things measured that **should not be
relied on**. This file is the forward-looking one: what we are doing next and
what would make it a real answer.

---

## The left rail: constraints

Hard, and not worth arguing with.

**One 4090, 24,564 MiB, shared with actual use.** The heaviest shipped config
peaks at 21,186 MiB. Nothing runs beside a render. `ncu` needs the card alone.

**Renders are the expensive thing.** Roughly 12 minutes at full length, so a
five-point sweep at two runs is about two hours. This is the binding
constraint on every quality question.

**Length: 362 is the ceiling** (`h3_rules.MAX_LENGTH`, the longest H3 was
trained on). Frame counts snap to a 17n+5 grid. The old "345 is the maximum"
claim was invented and is withdrawn -- do not reintroduce it. There is **no
fixed reference-video length**: OOM risk depends on reference count, size,
resolution and canvas, and no single frame count describes it.

**Canvas: the area cap means wider is free and buys nothing.** Every ratio at
or above 1.75 costs the same ~1008 tokens/frame. The only cheap direction is
toward square, on a smooth 32px ramp.

**Kernel:** Sol-Attn needs bf16, head_dim 128, sm_80+. Its eager reference is
O(T²) and refuses past 4 GiB, so it can never run at production length --
every correctness number is at a toy size unless something replaces it.

**No test suite.** Verification is a live ComfyUI plus a GPU. A code change
means a restart before it counts. `bench/smoke_h3.py` is the only thing that
submits a prompt; a green static validator on an unsubmitted graph is
unverified.

**Shared working tree.** Other sessions work here. Stage by path, never
`git add -A`, and check `git status` before committing.

---

## The right rail: what "good" means

The thing this project keeps getting wrong is not measurement, it is
**deciding in advance what would count as an answer**. So, per axis:

### Speed

**Good = a ratio between two arms that differ in one variable, at a length
above the token floor, on the workload actually run, with the build recorded.**

Every one of those clauses has failed here at least once. Ratios taken against
a baseline nobody ships (the retracted 1.611x), at a length that misrepresents
the config, on t2v when the work is reference-heavy, with an unrecorded
kernel build.

The floor matters specifically: Sol-Attn needs **roughly 60k tokens** before
it shows anything. Below that a null result reads as "this knob does nothing"
and is indistinguishable from a real negative.

### Accuracy and correctness

**Good = graded against an independent implementation, on inputs with the
right distribution, with a control that has been shown to fail.**

Synthetic `torch.randn` is not the right distribution: it gives a near-uniform
softmax, so a block-sparse router has nothing to find and the premise of the
method is absent. That single fact retired every accuracy figure this repo
used to quote.

A check is untrusted until it has gone red for the right reason. Prefer a
control the check compares against over asserting numbers computed in the test
itself.

### Quality

**Good = a person watched a clip end to end and said what they saw.**

There is no output-quality instrument here and building one is not on the
list. `docs/SOLATTN.md`'s Quality section is marked REOPENED because every
judgement in it came from grids of stills, and the failure mode that matters
-- a small persistent object dissolving over about four frames -- is temporal
and cannot be seen that way.

Numbers can rank kernels. Only watching can say whether any of it reached the
picture. **This is the axis nothing else substitutes for**, and it is the one
that costs the scarce resource.

### Provenance

**Good = the record says which artifact produced the number.**

The kernel build tag, the graph, the length, the canvas, the model path. The
fork build of `comfy_kitchen` declares a version identical to the stock wheel,
so without the local tag nothing distinguishes them.

---

## The dials, and where they sit

All in `workflows/h3_config.py`, all single switches, all regenerate with
`python workflows/build_workflows.py`.

| dial | default | range | cost effect |
|---|---|---|---|
| `CANVAS_TIER` | `full` | full / near / fast / draft | 1.00 / 0.91 / 0.73 / 0.58 attention |
| length | 362 | 17n+5 grid | linear in tokens |
| `REF_LORA_ENABLED` | `True` | True / False | LoRA path vs ref2va checkpoint |
| `REF_LORA_STRENGTH` | 1.0 | 0.0-1.0 | model interpolation |
| `SOL_CUDA_DEFAULTS` | Sol off in graphs | tau, window, sinks | see `docs/SOLATTN.md` |

**Iterate at `fast` (1152x768) and 243 frames; confirm at `full` and 362.**
`fast` is exact 3:2, 27% off attention, and only 0.25 of ratio from the
shipped canvas, so framing reads the same. At 243 frames it is 62,208 tokens
-- just above the Sol floor. `draft` is 55,296 and **below** it, so `draft` is
for "does the pipeline run", never for a Sol measurement.

---

## What is established

Evidence grade is inside each claim, because a trailing "(unverified)" is the
part that gets trimmed when a sentence is copied forward.

**Canvas cost is fully mapped.** MEASURED by enumerating `adapt_canvas` over
the legal ratio range: 95 distinct resolutions, every ratio at or above 1.75
costing an identical 1008 tokens/frame because the area cap binds and trades
width for height. Twenty legal landscape canvases sit between 1:1 and 16:9,
about 0.04 apart in ratio, and attention goes as the *square* of tokens. Three
of the four shipped tiers hit a common ratio exactly, which the default
(1.75) does not. All four verified as stable fixed points.

**The ref LoRA is structurally sound, and exact where it can be graded.**
MEASURED by `bench/analyze_ref_lora.py`. Coverage is exact: 474/474 modules,
zero unmatched either direction. Reconstruction splits on the **int8
boundary, not on rank** -- 64 non-quantized modules reconstruct at residual
0.0022 and cosine 1.0000, and those include the 51 rank-8 adaln projections
which are also the *most rewritten* modules in the model. The 200 int8 modules
are **not gradeable** from these artifacts: the delta is ~0.36 quantization
steps RMS and the only available target differences two independently
quantized checkpoints.

A naive read of those 200 says "the LoRA is wrong". **The three-way
cross-check refutes that.** We hold a second independent quantization
(`fp8_scaled`), so the two targets can be graded against each other: they
agree at cos 0.040, while the LoRA agrees with the fp8 target at 0.351 --
**9x better**. The LoRA is closer to the truth than either target, which is
exactly why neither can grade it. `int8_convrot` applies a rotation, so
differencing two independently rotated checkpoints yields the wrong quantity,
not merely a noisy one.

**Consequence:** the delta is comparable to the quantization step of the base
it is applied to, which bounds what *any* extraction can deliver. A
higher-rank LoRA would be writing detail the int8 checkpoint cannot store.

**The HF "hybrid" checkpoints are an adaln swap, and we can build them
locally.** MEASURED 2026-08-16 by byte-comparing
`smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models` against both parents, every
probe with a control confirming the parents actually differ there. The
community framing is "hybrid models"; the reality is narrower:

- Only `blocks.{N..49}.adaln_proj.linear.{weight,bias}` are taken from ref2va.
  **Everything else is 100% fl2va** -- attention, MLP, norms, `final_layer`.
  The filename range is which blocks take ref2va's adaln (`b20-49` = blocks
  20-49, boundary confirmed at 19/20).
- They are **already `int8_convrot`**, decoded from the file's own
  `comfy_quant` tensor. Same format and key set we run, so no conversion and
  no VRAM change -- and no bf16 version exists or can be derived from them.
- Net effect on delivered modulation is **1.1-4.7%**. The adaln *weight* is
  strongly anti-correlated between parents (cos -0.71 to -0.83) but the *bias*
  is cos ~0.9995 and dominates. It is a small broad perturbation, only
  ~1.3-2x more targeted at reference rows than at ordinary video rows.
- **It cannot be a reference-pathway transplant.** `adaln_proj`'s input is an
  8-dim timestep coordinate from the shared `adaln_t_table`; reference rows are
  distinguished only by being pinned at `VISUAL_COND_TIMESTEP`. It carries no
  reference image content.
- It deliberately leaves `final_layer.adaln_proj` on fl2va -- **the single
  most-rewritten tensor between the parents** (rel_delta 1.92).

**So it does coarsely, in four fixed steps and skipping the most-changed
tensor, what the ref LoRA does exactly and continuously across all 51 adaln
projections plus `final_layer`.** Both routes independently agree that adaln is
the only thing meaningfully different between the parents. Evidence behind the
community praise is three comments, no samples, no numbers.

**Not worth the 84 GB**: any variant is reproducible from files already on disk
by copying those tensors between two safetensors. The discrete block-selective
blend *is* a different interpolation path from a uniform LoRA strength, so it
remains a distinct experiment if anyone wants it -- just a local one.

**`subject_definitions` overrides the reference image. Two independent
instances.** MEASURED 2026-08-16, and this is the most transferable thing
learned this session because it is about prompting, not about a model.

`_ref_prompt()` emits a generic template asserting that Subject 2's
"**architecture**, palette, and lighting are carried into the target video".
The reference used here (`2-mountain_landscape.png`) is an alpine lake with
snow peaks, conifers and a wildflower meadow -- **no buildings, no structures,
nothing architectural**. Both renders put the man inside a building looking out
through a window. Neither could have got that from the image; both got it from
the definition. Confirmed independently by Gemini, blind, from the clips and
the reference alone.

That is the second instance of the pattern. The first, from the owner's own
prompting work, is a brunette reference described as blonde in
`subject_definitions` producing a blonde output despite `retention_analysis`
saying `fully_preserved`. Different attribute type, different session, one of
them accidental. **`subject_definitions` is binding authority and
`retention_analysis` cannot correct it.** A prompt that asserts an attribute
the reference lacks will get that attribute hallucinated rather than dropped.

The same prompt also specifies "steady interior room tone" over an outdoor
scene, and carries two contradictory lighting instructions. All three defects
come from one generic template pasted onto references it does not match.

**The LoRA and the ref2va checkpoint are equivalent on identity.** MEASURED
2026-08-16 by paired render: same graph, same seed, 243 frames, one variable
(fl2va + ref LoRA @1.0 against the ref2va checkpoint). Judged blind by the
owner and independently by Gemini, both without knowing which arm was which.

- **Identity: equivalent.** Face structure, hair, suit and tie all preserved
  against the reference in both. That is what the weight measurement predicted
  (all 51 adaln projections at cosine 1.0000) and it held.
- **Scene coherence: ref2va ahead, one pair.** Its background parallaxes
  correctly against the window frame during the camera move; the LoRA arm's
  reads as a flat backdrop. A temporal failure, invisible in stills.
- **Subject presence: 60% against 90% of runtime.** Largely explained by the
  prompt specifying no timing at all, so this is free variation rather than a
  model property.
- **Lighting: NOT evidence.** See the retraction in `docs/evidence.md`.

**Do not treat this as settling the question.** One pair, one seed, and a
prompt defective in three ways -- one of which demonstrably drove both outputs.
`REF_LORA_ENABLED` stays `True`; nothing here argues for flipping it.

**Attention is not very sparse on this workload.** MEASURED on captured
activations: at its most concentrated a query still needs 178 key blocks of
591 to hold 90% of its mass, and 394 at the first block. That bounds what any
block-sparse method can save here.

**Tensor-core rates on this box.** MEASURED, `bench/mma_rate.cu`: int8
m16n8k32 at 334.5 TMAC/s against bf16/f16 m16n8k16 f32-accumulate at 83.8
(0.25x) and f16-accumulate at 167.3 (0.50x). f32-accumulate forms issue at
half rate on sm_89, as upstream claims for sm_120.

---

## What is not established

| question | why it matters | blocker |
|---|---|---|
| Does fl2va+LoRA render the same as ref2va? | the LoRA is canonical on 18 graphs | **Rendered and judged 2026-08-16 -- partially answered, see below. Identity: equivalent. Scene coherence: ref2va ahead in one pair. Re-run owed on a sound prompt.** |
| Is the Sol exact kernel MMA-bound or staging-bound? | decides whether a 16-bit PV costs 2.5x or much less | an idle card, `ncu` |
| Sparsity error against quantization error, on real activations | if quantization is small, a 16-bit PV buys nothing | none but card time |
| Routed density at production length | nobody knows how much of Sol's work the exact branch is | the block probe, which does not exist yet |
| `start_percent` / `end_percent` at any length | zero measurements ever | card time |
| Anything on the reference-heavy workload | every Sol number is t2v on fl2va | one capture + reruns |

**The input gap deserves its own line.** Every Sol-Attn measurement in this
repo is t2v on the fl2va model with zero references, and the captured
activations are the same shape. But the work being done is reference-heavy.
Reference rows are pinned exact by `sink_conditioning`, so reference-heavy is
where Sol has the **least** room -- meaning every existing ratio is an
optimistic bound for the real workload. Closing this is one capture against
`h3_probe_sol_on_refs_api.json` with `H3_CAPTURE` set; `h3_capture.py` is
env-driven and needs no code change.

---

## Next, in order

**0. Fix `_ref_prompt()` before running anything else on references.** New top
priority as of 2026-08-16, and it displaces the Sol work because it is cheaper
and currently more load-bearing. The generator asserts attributes generically
-- "architecture, palette, and lighting" -- regardless of what the reference
actually contains, and that assertion is binding: both arms of the LoRA A/B
built a house that exists in no reference image. Every reference measurement in
this repo inherits the defect, and the fix needs no GPU. Either derive the
attribute list from the reference, or drop it and let the image speak.

Two further defects in the same template, same root cause: an interior
soundscape specified over an outdoor scene, and two contradictory lighting
instructions. Separately, all 20 reference prompts are single-shot with a
`detailed_description` of 46-73 words against the guide's stated 350-500, so a
ten-second clip is driven by about three seconds of instruction and the model
improvises the rest -- which is what produced the 60%-against-90% subject
presence in the A/B.

**0b. Re-run the LoRA A/B on a sound prompt.** Only after 0. Twenty-five
minutes of card time, and it is the actual answer to whether the LoRA is safe
as canonical. Today's run says equivalent on identity and leaves scene
coherence on one pair.

1. **Capture a reference-heavy render.** One render, no code change. Every
   measurement downstream is single-workload without it.
2. **Decompose Sol's error** on both captures (`bench/analyze_sol_error.py`,
   scaffolded, not implemented). Splits total error into sparsity against
   quantization. If quantization is small, the 16-bit PV question closes
   without a kernel being written. This is the gate that can end a whole line
   of work, so it runs before anything expensive.
3. **Port the block probe** (`sol_block_probe.py`, scaffolded, not
   implemented). Produces routed density and says which transformer blocks the
   sparsity is hurting. Unblocks `dense_blocks`, which is currently a guess.
4. **Profile the Sol stages** (`bench/profile_sol_stages.py`, scaffolded).
   Needs the card alone.
5. **Paired render**, fl2va+LoRA@1.0 against ref2va, same seed. The only thing
   that can close the reconstruction question.
6. **Watch a clip end to end.** Nothing above substitutes for it.

### Two decisions that need no card, added 2026-08-16

Both fell out of reading Sol-Engine's H3 profiles properly. Neither is
scheduled work; both are calls the owner can make.

**Adopt `dense_blocks="0-1"`, or decide not to.** Every H3 profile NVLabs
publishes runs the first two transformer blocks dense --
`SOL_ATTN_FIRST_DENSE_LAYERS=2` on the single-card RTX 5090 cell,
`H3_SOL_DENSE_LAYERS=2` on GB200, `dense_blocks: int = 2` on GB10, and the
A100 README's prose. We ship none, which makes their tested recipe strictly
more conservative than ours in exactly one place.

The cost, by arithmetic rather than measurement: the 2026-08-16 dense/sparse
pair was 860.8 s against 454.0 s across 50 blocks, so **assuming uniform
per-block cost** two blocks is about 16 s of 454, or 3.6%. That assumption is
the weak part and it is cheap to check with one arm.

What this does **not** need is the block probe at item 3. That instrument
answers "which blocks does sparsity hurt *here*", which is the question for
choosing **our own** list. Copying a list four hardware profiles already
validated needs no instrument, and conflating the two is why this sat as
`SOL_ARTIFACT_INSURANCE`, unwired, for weeks.

**Decide whether to build the NVLabs sm89 kernel and compare.** PR #464
(2026-08-15) added an official BF16 SM89 CuTe Sol-Attn kernel, so there are now
two independent 4090 implementations: comfy-kitchen's, which we build and run,
and NVLabs' own. Which is faster or more accurate on this card is unmeasured,
and it is the only external cross-check available to us since the Triton pack
was deleted.

Cost: one Python dependency (`cutlass.cute`; torch 2.13, CUDA 13.2 and Triton
3.7.1 already clear their floors) plus a seam, because their public API has no
`sink_q` and `exact_kv_and_rows`'s query half would have to be done outside the
kernel. Full accounting in [`docs/sol_upstream.md`](sol_upstream.md).

**Scaffolded means every entry point raises.** Three files exist to fix the
design decisions -- metric choice, the control each one needs, the permanent
node id -- before implementation, so those get argued once.

---

## Hazards that have each cost something here

- **A check that skips reads exactly like a check that passed.** Exit 2 is the
  convention for "did not run".
- **A green static validator on an unsubmitted graph is unverified.** Only the
  smoke submits.
- **Free the GPU before CUDA checks**, or they OOM and read as regressions.
- **Restart ComfyUI by port owner**, not `pgrep | head -1`, which kills the uv
  wrapper and leaves the old server serving stale `/object_info`.
- **`import nodes` resolves to ours.** Any script importing both must put
  ComfyUI's root first.
- **A running ComfyUI holds two copies of every `comfy_extras` module.**
  Resolve by identity, and verify against the live `/object_info`.
- **A measurement stated at a scope wider than it was taken at** is the single
  most common defect in this repo's history. When a claim is copied into a new
  sentence, the scope qualifier is the part that gets dropped.
- **Three arms agreeing is not a control.** A fourth arm that isolates the
  variable is. A 3.7 GB "saving" survived until the dense control showed the
  opposite sign.
