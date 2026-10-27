# Roadmap: what we are trying to find out, and what would count as finding it

Last updated: 2026-09-11.

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
canvas cost table, the Morton geometry, the checkpoint internals, the MMA rates
all needed zero or one render.

Two companion documents, and the split matters:
`docs/open_experiments.md` is the list of things **not measured** and why.
`docs/evidence.md` is the ledger of things measured that **should not be
relied on**. This file is the forward-looking one: what we are doing next and
what would make it a real answer.

## Closed lanes

Closed by the owner; propose nothing on one unless the owner reopens it.
History that still names them (the changelog, dated records, postmortems,
session logs) is deliberate and is not drift to clean up.

| lane | closed | what stands instead |
|---|---|---|
| the extracted fl2va-to-ref2va reference LoRA | 2026-08-21, `816f704` | reference graphs load the ref2va checkpoint; there is no second path |
| single-frame image generation and editing | 2026-08-27 | `docs/h3_image_editing.md` records what moved to `archive/`. Reference-image video graphs are live and not part of this |
| calibrating or quantising our own encoder (the llm-compressor AWQ/GPTQ lane) | 2026-08-27 | the encoder is `h3_config.MODELS["clip"]`; the lane's record was under `docs/research/qwen3-vl-special-tokens-post-training/` until 2026-10-12 and is in git history. **Closed after a badly executed attempt, not because the approach was refuted** (owner, 2026-09-20, `docs/wiki/decisions.md`); calibration data is the untapped part **`llmcompressor` removed as a dependency 2026-09-29 (owner: "llmcompressor is for awq stuff - remove that dependency")**: the nine bench files that imported it and their checks left in 0.175.0, the records stay. |
| PDD quality work | parked 2026-09-05 | "Owner decisions, 2026-09-05 evening" below, and `docs/research/pdd/2026-09-05_bake_plan.md` |
| turbo LoRAs: the lightx2v family (8-step, 768p v1.0 to v1.2, ref2v, Turbo-SLA) and any other "turbo" LoRA | 2026-09-26 (owner: "lightx and turbo stuff irrelevant"; the larryvrh pack went 2026-09-23) | the distill lanes are FastH3, PDD and, undecided, FlashGen. `check_distill_settings.py` fails any graph loading a turbo LoRA; the 14 graphs, the `TURBO_*` constants and their vendor-row grading are in git at 0.155.0 |
| SLA-style top-k selection in the Sol node (`selection` top-k, `keep_percent`, `pooled_tail`) | 2026-09-27 (owner: "SLA is closed") | adaptive tau. Kitchen's `topk_ratio` stays: VSA and core's `vsa` use it. Kitchen's top-k is SLA's selection without its learned linear branch, so no SLA-distilled LoRA was ever fully served; `docs/research/2026-09-27_sol_node_redesign.md` |
| Morton / token reordering for H3 (`morton`, `morton_curve`, `sol_curves.py`) | 2026-09-27 (owner: "Morton seems dead too") | native token order. No H3 implementation upstream reorders (NVLabs' GB200 H3 profile, LightX2V's refusal); core and kitchen have none. **Closed, not refuted:** `3d` still has lower Sol error at equal density on every captured cell (`bench/results/2026-09-17_sol_orderings.md`) |
| two or more subjects in one masked pass (`shot_to_load pair`, a two-subject load) | 2026-10-15 (owner: "lets kill multi person for now (again)", "or rather multi in one pass"), after `second_lead_0852` blended the two faces on playback | one load per subject; where two are close, one hole a pass: the second load is given the first's render as its video (`docs/wiki/decisions.md`, 2026-10-15). The pair tooling stays in the tree, unused |
| TaoMate-H3 (the streaming adapter as a ComfyUI LoRA, its probe graphs and the ported streaming sampler `MiniMaxH3TaoMateStreamSampler`) | 2026-09-27 (owner: "taomate is deprecated you can remove everything from it here. we're not gonna pursue it anymore.") | the distill lanes above. **Deprecated, not refuted.** The node, its runtime, graphs, checks and converter are in git before the removal; the records stay in `bench/results/` (`2026-09-15_taomate_*`); `docs/wiki/decisions.md` |
| open experiment #23, what INT8 costs at run time per module kind, and its Tier 1 activation observer (`MiniMaxH3QuantObserve`, `quant_observe.py`, `dit_observe.py`, `check_quant_observe.py`) | 2026-09-29 (owner: "experiment 23 is old i dont think anything has been done with it since august so retire everything related to that") | nothing measures activation rounding at run time. **Closed, not refuted:** the weight-side records stay in `bench/results/`, and `docs/open_experiments.md` #23 keeps its text as the record. Instrument and check are in git before the removal |
| our `MiniMaxH3VSAAttention` (FastVideo VSA as our own block-forward replacement) and its two probe graphs | 2026-09-29 (owner agreed with retiring it) | core's `BlockSparseAttention` in `vsa` mode, which every FastH3 graph uses. The node was parked and refused since 0.157.1; `docs/research/vsa/` is the record |
| `bench/restart_comfy.sh` | disabled 2026-09-02 | restart by hand, `docs/comfy_notes.md` |
| a hook that blocks pattern kills (`pkill`, `killall`) | tried and removed 2026-09-03 | kill one pid, found from the port owner |
| Sapiens matting for the masked lane (Meta's matting model composited at the hair edge; `internal/gemini/2026-10-17_brief_build_matting_relay.md`) | 2026-10-18 (parked at about 13:53Z, closed at about 21:42Z) | the pack's own per-piece mattes under one compositing law (`bench/assemble_delivery.py::lay_plane`). **Closed, not refuted:** never built into the chain |
| SAM 3 base in the masked lane's tracking (in place of SAM 3.1) | 2026-10-18 (parked earlier in the day, closed at about 21:42Z) | SAM 3.1 with memory selection, on in captures; the node leaves SAM 3 base as ComfyUI has it. **Closed, not refuted:** on one face test (0613g, 680-760) base lost 61 frames and came back early on her hair (`docs/wiki/decisions.md`, 2026-10-18). Its converted checkpoint, `models/checkpoints/sam3_fp32.safetensors`, is the owner's to delete |
| PDMD (pdmd2026's 4-step and 2-step LoRAs, through kijai's resizes) | parked 2026-10-01 (owner: "nothing else to do with it"); retired 2026-10-05 (owner: "lets retire it. i didnt see anything from it worth keeping") | Nothing: its graphs, the `PDMD_*` constants, the check cases, the conversion script, the arm manifest and `docs/research/pdmd/` are out of the tree and in git. `bench/check_distill_settings.py` fails a graph that loads a PDMD LoRA. **Retired, not refuted:** one blind look, in which FlashGen was preferred and PDMD ghosted at shot transitions (`data/results/2026-10-01_2026-10-01_pdmd_vs_flashgen_verdict.json`). The LoRA files under ComfyUI's `models/loras/h3/` and the `coderef/pdmd` checkout are the owner's to delete |

## Historical forward plans (September 2026)

> Archived verbatim to [`docs/wiki/archive/2026-09-05_roadmap_forward_plans.md`](wiki/archive/2026-09-05_roadmap_forward_plans.md) (`2026-09-03` forward plan, `2026-09-04` forward plan, and `2026-09-05` owner scoring decisions). For the live queue, see [`docs/wiki/handover.md`](wiki/handover.md) and [`docs/wiki/next_steps.md`](wiki/next_steps.md).

## The left rail: constraints

Hard, and not worth arguing with.

**One 4090, 24,564 MiB, shared with actual use.** The heaviest shipped config
peaks at 21,186 MiB. Nothing runs beside a render. `ncu` needs the card alone.

**Renders are the expensive thing, and there are now two regimes.** The base
model at 16 steps is roughly 12 minutes at full length, so a five-point sweep
at two runs is about two hours. A 4-step lightx2v student at the same canvas
and length was 169 s end to end on 2026-08-20
(`data/results/2026-08-20_sla_arms.jsonl`, at a 330 W power limit), which is
what makes the distribution standard below affordable. This is the binding
constraint on every quality question, and which regime you are in decides
what a day can hold.

**Length: 362 is the ceiling** (`h3_rules.MAX_LENGTH`, the longest H3 was
trained on). Frame counts snap to a 17n+5 grid. No shorter maximum exists; do not
introduce one. A window is as long as its shot needs: attention is quadratic. There is **no
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
right distribution.**

Synthetic `torch.randn` is not the right distribution: it gives a near-uniform
softmax, so a block-sparse router has nothing to find and the premise of the
method is absent. That single fact retired every accuracy figure this repo
used to quote.

Prefer a control the check compares against over asserting numbers computed
in the test itself.

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
| `SOL_RECOMMENDED_CUDA` / `SOL_PDD_CUDA` / `SOL_CUDA_DEFAULTS` | Sol **on** in every video graph since 2026-08-14; `tau` 1.0 from 2026-08-20 by owner decision. **Two shipped configs from 2026-08-29, one again since 2026-09-11**, when `end_percent` moved to 1.0 on every graph, adopting upstream (sglang's `sol_attn` and core's `BlockSparseAttention` agree); `h3_config.sol_for_graph` is still the resolver | tau, window, sinks, dense blocks | see `docs/SOLATTN.md`; 1.3 returns only if it shows no difference from 1.0 on the distilled LoRAs while buying meaningful speed. The PDD split is an owner reading of rendered arms, four knobs moved together, so nothing in it attributes an effect to one knob -- a blind distribution per knob is what would |
| `CACHE_NODE` | probe graphs only, **not canonical** (owner decision 2026-08-20) | EasyCache threshold/window | 1.56-1.74x on deterministic samplers at 16 steps; a 16-step lever with nothing to skip at 4 |

**The PDD Sol recipe is four eyeballed knobs, and two of them have a
calculable answer waiting.** `docs/SOLATTN.md`, "What would replace the
eyeballing, knob by knob", carries the derivations and the three things that
would settle them, cheapest first: (1) `end_percent` 0.74 against 0.87 at 8
evaluations (moot since 2026-09-11, when every graph moved to 1.0), a single-axis test of whether Sol's dense region is pinned to the
sigma path or to the schedule -- the recipe already bets on the first; (2)
`bench/analyze_sol_error.py` at tau 1.0 over the surviving fl2va capture, which
ranks blocks by Sol's own error on the partition the t2v PDD arms run and needs
no render; (3) a timed A/B for what `dense_blocks` costs, which is legal
because timing is not a sample. The measured ranking currently says block 0 is
where Sol is MOST accurate and block 40 where it is worst, which is evidence
against the shipped `0,1,2` and for a set nobody ships.

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

  > **2026-08-20: the anti-correlation is an artifact, not a real feature
  > the bias happens to mask.** The two checkpoints' `adaln_t_table` bases
  > agree in sign on columns 0-3 and are sign-flipped on columns 4-7
  > (per-column cosine +1, +1, +0.996, +0.996, -0.9997, -0.9997, -0.99,
  > -0.99), and the coefficient columns on the flipped basis columns are
  > the large-norm ones. Compared at the modulation output, where the
  > basis is applied, the *time-varying* part of the modulation differs
  > 5-9% per block with cosine 0.996-0.999, and the whole output 1.4-4.7%
  > -- the same 1.1-4.7% this bullet measured, now without a mechanism
  > that was wrong. `data/results/2026-08-20_dit_internals.json`.
- **It cannot be a reference-pathway transplant.** `adaln_proj`'s input is an
  8-dim timestep coordinate from the shared `adaln_t_table`; reference rows are
  distinguished only by being pinned at `VISUAL_COND_TIMESTEP`. It carries no
  reference image content.
- It deliberately leaves `final_layer.adaln_proj` on fl2va -- ~~the single
  most-rewritten tensor between the parents (rel_delta 1.92)~~ **withdrawn
  2026-08-20**: same coefficient-level artifact as above. At the modulation
  output the final layer's time-varying part differs ~12%, the largest
  single adaln item but not "rewritten".

**So it swaps adaln in four fixed steps.**
~~Both routes independently agree that adaln is the only thing meaningfully
different between the parents.~~ **Withdrawn 2026-08-20.** The int8 linears
differ ~3.2% relative between the parents and the modulation output 1.4-4.7%;
nothing at the weight level singles adaln out, and the hybrids keep fl2va's
linears, which differ from ref2va's by as much as the adaln they swap. The
2026-08-16 byte-comparison and the 1.1-4.7% figure stand; the reading that
adaln was *the* difference does not. Evidence behind the community praise is
three comments, no samples, no numbers.

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
| Sparsity error against quantization error, on real activations | if quantization is small, a 16-bit PV buys nothing | **Done 2026-08-17/19**, item 2 below; quantization is not small, and on 2026-08-20 it was found to track Q/K magnitude rather than V (`docs/open_experiments.md` #17b) |
| Routed density at production length | nobody knows how much of Sol's work the exact branch is | **Done 2026-08-19**, `bench/sweep_routing_density.py`; flat per block and per step, structured per head, priced empty |
| `start_percent` / `end_percent` at any length | zero measurements ever | card time |
| Anything on the reference-heavy workload | every Sol number is t2v on fl2va | **Closed 2026-08-17**: four reference-heavy captures now exist (2026-08-17 fl2va+ref LoRA, its multistep sibling, 2026-08-18 ref2va, 2026-08-20 clean fl2va), and every error number since is on them |
| What the 330 W power limit costs | decides whether 330 W timings compare to the 450 W records, and the bandwidth-bound reading in `docs/hardware.md` | one `sudo`; run 2026-08-20, `bench/results/2026-08-20_power_limit_pair.jsonl` -- verdict below when written |

**Power, stated once.** No 330-vs-450 W pair was ever run before 2026-08-20; it
was a forward item three times and blocked on sudo each time. The 18th's
records are at 450 W, the 2026-08-20 morning's at 330 W, and ratios within a
day hold while absolute seconds across that boundary do not. The decision rule
written on 2026-08-17: a delta under ~2% supports the bandwidth-bound reading;
a delta near the clock delta supports L2-bound.

**Run 2026-08-20** on the 4-step 768p turbo graph (fl2va + v1.0 LoRA,
1344x768 x 362, Sol at tau 1.0), two timed runs per limit with disjoint seed
bases, `bench/results/2026-08-20_power_limit_pair.jsonl` and its verdict
sidecar `_verdict.json`: sampler 134.1 s at 330 W against 126.8 s at 450 W,
within-arm spread 0.2%, so the cap costs **5.8%** -- against a **12.5%**
core-clock delta (p50 2145 against 2452 MHz during sampling, power pegged at
328 and 446 W). Between the two readings: sampling at this workload is partly
core-clock-bound, neither cleanly bandwidth- nor L2-bound. Stated for this
4-step t2v workload; the 16-step all-refs pair remains unrun.

---

## Historical regime question, token ordering & August 2026 queue

> Archived verbatim to [`docs/wiki/archive/2026-08-20_roadmap_regime_and_ordering.md`](wiki/archive/2026-08-20_roadmap_regime_and_ordering.md) (`The regime question (opened 2026-08-20)`, `CLOSED 2026-08-16: token ordering as a quality question`, and the August 2026 error-decomposition queue).

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
- **A deterministic arm re-run at a held seed is a node-cache hit**, and two
  arms that share a graph file need disjoint seed bases. ComfyUI returns the
  cached output for a byte-identical resubmission in 0.0 s, which reads as a
  fast render. `bench/run_graph_arms.py` refuses `--runs > 1` without `--seed`
  and flags `suspect_cache_hit` for this reason; a power-limit pair on one
  graph nearly shipped as two cache hits on 2026-08-20.
