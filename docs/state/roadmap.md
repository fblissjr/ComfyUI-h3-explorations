# Roadmap: what we are trying to find out, and what would count as finding it

Last updated: 2026-10-19 (the superseded forward plans of 2026-09-03 and 2026-09-04 cut to where they stand; their
full text is in git at `901bee1`, `docs/state/roadmap.md`).

## What this is, and who it is for

A working plan for the H3 speed-and-quality investigation, written so a fresh session can pick it up without
re-deriving the constraints or re-litigating decisions already made. One person tinkering on a single RTX 4090 with
24 GB: no team, no cluster, no deadline. **Card time is the scarce resource and rigour is cheap**: a minute of Python
that settles something beats an afternoon of renders that yields a number nobody can defend.

Companions: [`open_experiments.md`](open_experiments.md) is what is **not measured** and why;
[`../method/evidence.md`](../method/evidence.md) is what was measured and **should not be relied on**;
[`next_steps.md`](next_steps.md) is the short list of what to do next. The masked video-to-video lane's order is the
day's [`handover.md`](handover.md) and `data/AGENTS.md`, not this page.

## Closed lanes

Closed by the owner; propose nothing on one unless the owner reopens it.
History that still names them (the changelog, dated records, postmortems,
session logs) is deliberate and is not drift to clean up.

| lane | closed | what stands instead |
|---|---|---|
| the extracted fl2va-to-ref2va reference LoRA | 2026-08-21, `816f704` | reference graphs load the ref2va checkpoint; there is no second path |
| single-frame image generation and editing | 2026-08-27 | `docs/h3/image_editing.md` records what moved to `archive/`. Reference-image video graphs are live and not part of this |
| calibrating or quantising our own encoder (the llm-compressor AWQ/GPTQ lane) | 2026-08-27 | the encoder is `h3_config.MODELS["clip"]`; the lane's record was under `docs/research/qwen3-vl-special-tokens-post-training/` until 2026-10-12 and is in git history. **Closed after a badly executed attempt, not because the approach was refuted** (owner, 2026-09-20, `docs/state/decisions.md`); calibration data is the untapped part **`llmcompressor` removed as a dependency 2026-09-29 (owner: "llmcompressor is for awq stuff - remove that dependency")**: the nine bench files that imported it and their checks left in 0.175.0, the records stay. |
| PDD quality work | parked 2026-09-05 | the owner, 2026-09-05 evening: PDD8 lost to the sage floor on every scene under sage alone and the bake did not move it, so the look is the schedule's ("Owner decisions, 2026-09-05 evening" below; `docs/research/pdd/2026-09-05_bake_plan.md`). The bake stays as the PDD8 checkpoint on hygiene alone |
| turbo LoRAs: the lightx2v family (8-step, 768p v1.0 to v1.2, ref2v, Turbo-SLA) and any other "turbo" LoRA | 2026-09-26 (owner: "lightx and turbo stuff irrelevant"; the larryvrh pack went 2026-09-23) | the distill lanes are FastH3, PDD and, undecided, FlashGen. `check_distill_settings.py` fails any graph loading a turbo LoRA; the 14 graphs, the `TURBO_*` constants and their vendor-row grading are in git at 0.155.0 |
| SLA-style top-k selection in the Sol node (`selection` top-k, `keep_percent`, `pooled_tail`) | 2026-09-27 (owner: "SLA is closed") | adaptive tau. Kitchen's `topk_ratio` stays: VSA and core's `vsa` use it. Kitchen's top-k is SLA's selection without its learned linear branch, so no SLA-distilled LoRA was ever fully served; `docs/research/2026-09-27_sol_node_redesign.md` |
| Morton / token reordering for H3 (`morton`, `morton_curve`, `sol_curves.py`) | 2026-09-27 (owner: "Morton seems dead too") | native token order. No H3 implementation upstream reorders (NVLabs' GB200 H3 profile, LightX2V's refusal); core and kitchen have none. **Closed, not refuted:** `3d` still has lower Sol error at equal density on every captured cell (`bench/results/2026-09-17_sol_orderings.md`) |
| two or more subjects in one masked pass (`shot_to_load pair`, a two-subject load) | 2026-10-15 (owner: "lets kill multi person for now (again)", "or rather multi in one pass"), after `second_lead_0852` blended the two faces on playback | one load per subject; where two are close, one hole a pass: the second load is given the first's render as its video (`docs/state/decisions.md`, 2026-10-15). The pair tooling stays in the tree, unused |
| TaoMate-H3 (the streaming adapter as a ComfyUI LoRA, its probe graphs and the ported streaming sampler `MiniMaxH3TaoMateStreamSampler`) | 2026-09-27 (owner: "taomate is deprecated you can remove everything from it here. we're not gonna pursue it anymore.") | the distill lanes above. **Deprecated, not refuted.** The node, its runtime, graphs, checks and converter are in git before the removal; the records stay in `bench/results/` (`2026-09-15_taomate_*`); `docs/state/decisions.md` |
| open experiment #23, what INT8 costs at run time per module kind, and its Tier 1 activation observer (`MiniMaxH3QuantObserve`, `quant_observe.py`, `dit_observe.py`, `check_quant_observe.py`) | 2026-09-29 (owner: "experiment 23 is old i dont think anything has been done with it since august so retire everything related to that") | nothing measures activation rounding at run time. **Closed, not refuted:** the weight-side records stay in `bench/results/`, and `docs/state/open_experiments.md` #23 keeps its text as the record. Instrument and check are in git before the removal |
| our `MiniMaxH3VSAAttention` (FastVideo VSA as our own block-forward replacement) and its two probe graphs | 2026-09-29 (owner agreed with retiring it) | core's `BlockSparseAttention` in `vsa` mode, which every FastH3 graph uses. The node was parked and refused since 0.157.1; `docs/research/vsa/` is the record |
| `bench/restart_comfy.sh` | disabled 2026-09-02 | restart by hand, `docs/working/comfy_notes.md` |
| a hook that blocks pattern kills (`pkill`, `killall`) | tried and removed 2026-09-03 | kill one pid, found from the port owner |
| Sapiens matting for the masked lane (Meta's matting model composited at the hair edge; `internal/gemini/2026-10-17_brief_build_matting_relay.md`) | 2026-10-18 (parked at about 13:53Z, closed at about 21:42Z) | the pack's own per-piece mattes under one compositing law (`bench/assemble_delivery.py::lay_plane`). **Closed, not refuted:** never built into the chain |
| SAM 3 base in the masked lane's tracking (in place of SAM 3.1) | 2026-10-18 (parked earlier in the day, closed at about 21:42Z) | SAM 3.1 with memory selection, on in captures; the node leaves SAM 3 base as ComfyUI has it. **Closed, not refuted:** on one face test (0613g, 680-760) base lost 61 frames and came back early on her hair (`docs/state/decisions.md`, 2026-10-18). Its converted checkpoint, `models/checkpoints/sam3_fp32.safetensors`, is the owner's to delete |
| PDMD (pdmd2026's 4-step and 2-step LoRAs, through kijai's resizes) | parked 2026-10-01 (owner: "nothing else to do with it"); retired 2026-10-05 (owner: "lets retire it. i didnt see anything from it worth keeping") | Nothing: its graphs, the `PDMD_*` constants, the check cases, the conversion script, the arm manifest and `docs/research/pdmd/` are out of the tree and in git. `bench/check_distill_settings.py` fails a graph that loads a PDMD LoRA. **Retired, not refuted:** one blind look, in which FlashGen was preferred and PDMD ghosted at shot transitions (`data/results/2026-10-01_2026-10-01_pdmd_vs_flashgen_verdict.json`). The LoRA files under ComfyUI's `models/loras/h3/` and the `coderef/pdmd` checkout are the owner's to delete |

## The attention plan, where it stands

The goal, in the owner's words: personal experiments, shared with the community, to find the best quality and
performance this box can render. Operationally: how much time sparse attention and the distills can take out of an
H3 render before the result visibly degrades, per scene, measured from the true baseline
(`workflows/bench/h3_text_to_video_dense_stamped_api.json`: stock attention, no LoRA), with the owner's blind judgment
as the top-level measure and the numeric proxies reported beside it.

The shipped chain is Sol inside core's Model Attention Backend on kitchen int8, sage off (2026-09-15,
[`decisions.md`](decisions.md)); it replaced the 2026-09-04 "sage is always on" floor. The routing policy (where Sol
runs: which steps, blocks and rows) was set as the core priority on 2026-09-04. Its steps, as they stand:

| step | state | where |
|---|---|---|
| probe the subway scene against standoff | done 2026-09-04: same ranking, the slip was the floor's | `data/results/2026-09-04_sol_probe_subway_vs_standoff.json` |
| the PDD ladder | judged 2026-09-05; PDD8's loss is PDD's own; PDD parked | "Closed lanes" above |
| segment policy: text rows dense | adopted 2026-10-02 for every graph (`sink_conditioning` `exact_kv_and_all_rows`, 0.185.0) | `../../bench/results/2026-10-02_sol_dense_blocks_panel.md` |
| block policy | `tau_profile` retired 2026-09-27; `dense_blocks` is Option C since 2026-10-02, and does not move the owner's verdict | `h3_config.SOL_CUDA_DEFAULTS`; [`decisions.md`](decisions.md), 2026-10-02 |
| step policy: the window's start | open | [`open_experiments.md`](open_experiments.md) #27 |
| the other populations (legal PDD8 through the node's own sigmas, Ref2VA), each ranked on its own | not run | the probe, `sol_block_probe.py` |
| token routing (`token_aug`) | graded; ships off; its output is not reproducible on block 49, cause measured | [`open_experiments.md`](open_experiments.md) #29; [`../research/2026-09-05_token_aug_plan.md`](../research/2026-09-05_token_aug_plan.md) |

The output check of 2026-10-03 closed the search for a proxy: no measurement predicts the owner's verdict, and the
shipped default stays (`../../bench/results/2026-10-03_sol_output_check.md`).

## Owner decisions, 2026-09-05 evening

`decisions.md` starts at 2026-09-11, so these dated decisions, with the two of 2026-09-04 they followed, live here. Each with what became of it.

| date | decision | now |
|---|---|---|
| 2026-09-04 | **Sage is always on**: the floor every comparison is judged against is the sage render; the dense stamped arm states what the model itself does, rendered when asked | reversed 2026-09-15: kitchen int8 under Sol, sage off ([`decisions.md`](decisions.md), 2026-09-15) |
| 2026-09-04 | **The routing policy is the core priority**, measured on the online probe (`sol_block_probe.py`); the PDD ladder drops its stock-attention control; the fp16-sage rung is dropped (slower than Sol on every scene); no more seeds for now | the plan table above |
| 2026-09-05 evening | 1. **Composition differences count as prompt adherence** | stands: [`../method/eval_comparison.md`](../method/eval_comparison.md), judging |
| 2026-09-05 evening | 2. **PDD is parked**: PDD8 lost to the sage floor on every scene under sage alone, and the bake did not move it; no partition arm | stands ("Closed lanes"); the bake stays on hygiene alone |
| 2026-09-05 evening | 3. **The turbo rung goes**, two LoRAs under sage alone, blinded against the floor | closed unscored with the turbo lane, 2026-09-26 |
| 2026-09-05 evening | 4. **Token routing stage one goes** | done 2026-09-08 (#29) |
| 2026-09-05 evening | 5. **Two seeds** for the turbo rung, superseding 2026-09-04's "no more seeds" | moot with the turbo lane |

**The decision standard.** Trustworthy directional evidence promotes an EXPERIMENT. A shared workflow default
requires set-level testing, clean timing in an uninstrumented process, and a small blind multi-scene comparison;
`dense_blocks="0-2,32"` became a default without those and was withdrawn, and that is the precedent this sentence
exists to break.

**What would count as finding it:** a setting that survives a blind multi-scene comparison on more than one scene and
seed, with its cost measured in a clean process, and its record naming the prompt, seed, canvas, length and build.

## The left rail: constraints

Hard, and not worth arguing with.

**One 4090, 24,564 MiB, shared with actual use.** The heaviest shipped config
peaks at 21,186 MiB. Nothing runs beside a render. `ncu` needs the card alone.

**Renders are the binding cost, in two regimes.** The base model at 16 steps
is about 12 minutes at full length (a five-point sweep at two runs is about two
hours). A 4-step distilled graph at the same canvas and length was 169 s end to
end on 2026-08-20 (a lightx2v student, a lane since closed;
`data/results/2026-08-20_sla_arms.jsonl`, at 330 W). Which regime you are in
decides what a day can hold.

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

**Shared working tree.** Other sessions work here; the git rules are root `AGENTS.md`, "The shared tree".


## The right rail: what "good" means

Decide in advance what would count as an answer; that, not measurement, is what this project keeps getting wrong.

| axis | good means | the failure it guards against |
|---|---|---|
| speed | a ratio between two arms that differ in one variable, above the token floor, on the workload actually run, with the build recorded | ratios against a baseline nobody ships (the retracted 1.611x), at a misleading length, on t2v when the work is reference-heavy, with an unrecorded kernel build. Sol needs roughly 60k tokens before it shows anything; below that a null reads as "this knob does nothing" |
| accuracy | graded against an independent implementation, on inputs with the right distribution; a control the check compares against, not numbers computed in the test | `torch.randn` gives a near-uniform softmax, so a block-sparse router has nothing to find; that retired every accuracy figure this repo used to quote |
| quality | a person watched a clip end to end and said what they saw | grids of stills (why `../attention/sol_attn.md`'s Quality section is REOPENED): a small object dissolving over about four frames is temporal. No output-quality instrument exists ([`open_experiments.md`](open_experiments.md) #14). Numbers rank kernels; only watching says whether it reached the picture |
| provenance | the record names the artifact that produced the number: kernel build tag, graph, length, canvas, model path | the `comfy_kitchen` fork declares the stock wheel's version, so without the local tag nothing tells them apart |

## The dials, and where they sit

All in `workflows/h3_config.py`, all single switches, all regenerate with `python workflows/build_workflows.py`.

| dial | default | range | cost effect |
|---|---|---|---|
| `CANVAS_TIER` | `full` | full / near / fast / draft | 1.00 / 0.91 / 0.73 / 0.58 attention |
| length | 362 (`h3_rules.MAX_LENGTH`) | 17n+5 grid | linear in tokens; attention quadratic |
| `SOL_RECOMMENDED_CUDA` / `SOL_PDD_CUDA` / `SOL_CUDA_DEFAULTS` | Sol on in every video graph (`bench/check_attention_defaults.py::SOL_EXEMPT_STEMS` is the exempt set); `tau` 1.0 since 2026-08-20; one config for every graph since 2026-09-11 (`end_percent` 1.0, adopting upstream); `h3_config.sol_for_graph` is the resolver | tau, window, sinks, dense blocks | see [`../attention/sol_attn.md`](../attention/sol_attn.md) |
| `CACHE_NODE` | probe graphs only, not canonical (owner, 2026-08-20) | EasyCache threshold/window | 1.56-1.74x on deterministic samplers at 16 steps; nothing to skip at 4 |

**Iterate at `fast` (1152x768) and 243 frames; confirm at `full` and 362.** `fast` is exact 3:2, 27% off attention,
and 0.25 of ratio from the shipped canvas, so framing reads the same. At 243 frames it is 62,208 tokens, just above
the Sol floor. `draft` is 55,296 and **below** it, so `draft` is for "does the pipeline run", never for a Sol
measurement.

## What is established

Evidence grade is inside each claim, because a trailing "(unverified)" is the part that gets trimmed when a sentence
is copied forward. The full reasoning of each is in git at `901bee1`.

- **Canvas cost is fully mapped** (MEASURED, `adapt_canvas` over the legal ratio range): 95 distinct resolutions;
  every ratio at or above 1.75 costs the same 1008 tokens/frame, because the area cap binds; twenty legal landscape
  canvases between 1:1 and 16:9 about 0.04 apart in ratio ([`../attention/morton.md`](../attention/morton.md)).
- **The HF "hybrid" checkpoints are an adaln swap we can build locally** (MEASURED 2026-08-16 by byte comparison):
  only `blocks.{N..49}.adaln_proj.linear` comes from ref2va, everything else is fl2va, already `int8_convrot`; the net
  effect on delivered modulation is 1.1-4.7% (`data/results/2026-08-20_dit_internals.json`). It is not a
  reference-pathway transplant (`adaln_proj` reads only an 8-dim timestep coordinate). The 2026-08-16 readings that
  the weight anti-correlation was real and that adaln was *the* difference between the parents were withdrawn
  2026-08-20. Not worth the 84 GB: any variant is a tensor copy between two files on disk.
- **`subject_definitions` overrides the reference image** (MEASURED 2026-08-16, two instances): a prompt that asserts
  an attribute the reference lacks gets it hallucinated, and `retention_analysis` cannot correct it.
- **Attention is not very sparse on this workload** (MEASURED on captured activations): at its most concentrated a
  query still needs 178 key blocks of 591 to hold 90% of its mass, and 394 at the first block.
- **Tensor-core rates on this box** (MEASURED, `bench/mma_rate.cu`): int8 m16n8k32 334.5 TMAC/s; bf16/f16
  f32-accumulate 83.8 (0.25x); f16-accumulate 167.3 (0.50x).
- **The 330 W cap costs 5.8% of sampler time against a 12.5% clock delta** on the 4-step workload; 450 W stays
  ([`../working/hardware.md`](../working/hardware.md); `bench/results/2026-08-20_power_limit_pair.jsonl`).
- **Per-block, per-step and per-head structure is real and not exploitable at those granularities**: routed density
  is flat per block and per step (`bench/sweep_routing_density.py`); a per-head dense escape loses to the global tau
  (`data/results/2026-08-19_head_granularity_arms.json`, `bench/price_head_arms.py`). Anything proposing per-head work
  answers that table first.
- **Speaker attribution across a cut**: asked for a singer before a cut and two speakers after, 11 of 12 clips hand the
  singing to a character after the cut, across selections and LoRAs: a model-and-prompt finding
  (`data/results/2026-08-22_subway_speaker_bleed.json`).
- **The audio hum** is the prompt asking for continuous texture, amplified by 4-step distillation, not Sol or the
  kernel (`data/results/2026-08-22_audio_hum.json`).
- **Block 49's loud K channels** (82, 34, 67, 19) are a weights property of the release, in both checkpoints
  (`data/results/2026-08-20_head_magnitudes.json`); what was done about them is
  [`../attention/block49_quant_error.md`](../attention/block49_quant_error.md).

## What is not established

| question | why it matters | blocker |
|---|---|---|
| Does fl2va+LoRA render the same as ref2va? | the LoRA was canonical on 18 graphs (the lane closed 2026-08-21) | judged 2026-08-16 on one pair: identity equivalent, scene coherence ref2va ahead; a re-run on a sound prompt owed |
| `start_percent` at any value | the window's start | [`open_experiments.md`](open_experiments.md) #27 |
| Whether a 16-bit PV pays on the CUDA Sol kernel | quant error is a measurable share | [`open_experiments.md`](open_experiments.md) #17 |
| The 16-step all-refs power pair | whether the 4-step reading of the power cap holds at 16 steps | optional, one `sudo` |
| An NVLabs sm89 Sol kernel against comfy-kitchen's | the only external cross-check on this card | their API has no `sink_q`; [`../attention/sol_upstream.md`](../attention/sol_upstream.md) |

If the error-decomposition line is resumed: compare the two routing MASKS, not two outputs (chunking the oracle to
production length was refuted 2026-08-19: it goes red on tie flips while both implementations are correct;
`data/results/2026-08-19_oracle_gate_scaling.json`).

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

