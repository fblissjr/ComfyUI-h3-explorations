# What holds, and what does not

Last updated: 2026-09-03; the encoder and reference-sizing rows 2026-09-13.

`docs/checks.md` indexes what is *checked*. `docs/open_experiments.md` indexes
what is *not measured*. This file is the third case and the one that kept
biting: claims that **were** measured, are written down as numbers, and should
not be relied on — because the conditions they were taken under are not the
conditions we ship.

An entry leaves this file when someone re-measures it, not when someone
re-reads it.

## Relationship to `SOLATTN.md`'s header

Both were written on 2026-08-14, in parallel, by two sessions that did not
know the other was doing it — which is its own entry in the postmortem.

The split, so neither drifts: **SOLATTN.md's header is the Sol-Attn quick
reference**, read by someone about to quote a number off that page. **This
file is the repo-wide ledger** and owns the three things a topic page cannot:
claims that are not about Sol-Attn (the fp8/fp16 accuracy figure is a sage
claim; the smoke's prompt substitution is a harness one), the **what restores
it** column, and the environment section below — the venv and `coderef/` are
not in git and nothing else records what changed there.

If the two disagree, this file is wrong until proven otherwise: SOLATTN.md is
maintained by whoever is running the measurements.

## Why this file exists

On 2026-08-14 two sessions working the same repo found nine separate claims in
this class, and **not one of them was caught by whoever wrote it**. Every one
came from a second reader. The pattern was identical each time: a number
measured under one configuration, carried into another, with the qualifying
sentence living in a different file — or in the same file, one paragraph away
from the table.

That is caveat decay, and `docs/checks.md` already names it. What it did not
have was somewhere for the *surviving* list to live, so each caveat was
attached to whichever page happened to discuss it, and none of them were
attached to the number.

---

## Do not rely on these

| claim | why not | what restores it |
|---|---|---|
| **"the six-section prompt format protects the speech in a reused soundtrack", and...** | Never established | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **the 2026-08-22 selection run reporting that top-k helps the SLA LoRA and hurts...** | VOID, same day, by replication. | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **any timing measured on a long-lived ComfyUI session** | Generalised from the row above on the day it happened, so treat it as a... | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **any capture manifest's attention.sol_attn, spelled here once as...** | FIXED 2026-08-17, same day | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **any capture manifest's attention.sage_mode, models.clip, and any models.loras...** | FIXED 2026-08-26, same day. | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **Sol vs sage = 1.611x** | Wrong on two axes, not three | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **centroid_tail = 2.5% e2e** | Ours, two runs, 0.1% spread, at 362 | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **Upstream's centroid_tail ~5–10% e2e** | Upstream conversation, his box, his settings. | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **CUDA is 1.4x over Triton e2e** | Upstream conversation. | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **centroid_tail is "~1.4x", or "the biggest speed knob in the node"** | Tripwire row, and the trap has now caught two readers. | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **reuse_qkv_memory saves nothing** | Not a negative result. | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **any peak-VRAM figure between 13:08:49 and f1dff99** | An external write reverted the device poller on disk | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **any fp8-vs-fp16 sage accuracy ratio, spelled here once as "2.7x more accurate"...** | WITHDRAWN 2026-08-16 by the owner, and deleted rather than caveated | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **"the LoRA arm carried the reference's studio lighting onto his cheek"** | Retracted 2026-08-16, same day, by the author. | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **any reference measurement taken before _ref_prompt() is fixed** | RESTORED 2026-08-17 | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **"AdaLN is replaced, not adjusted, in every block" / final_layer.adaln_proj is...** | Withdrawn 2026-08-20. (essentially rewritten) | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **Sol is 0.999919 accurate** | Implementation fidelity, not total error | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **text = 38 rows, sequence = 12,264** | smoke_h3.py substitutes both the prompt... | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **everything derived from that** | all smoke-harness statements. | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **the sink's audio framing** | A t2v framing. | [detail](wiki/archive/2026-08-22_retracted_claims_part1.md) |
| **reference-load table** | Wrong on three axes (23-point swing) | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **"with Sol on, sage gets nothing"** | Retracted 2026-08-14. | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **min_tokens 4096 is "very likely wrong"** | Retracted, and the retraction's reasoning was itself corrected 2026-08-14 | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **"fp16 lands on the steps where precision matters most"** | Inference, not measurement. | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **any bench progress read from its own stdout mid-run** | The warmup print lacks flush=True, and tee makes stdout block-buffered, so... | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **Morton is "worth 1.16x alone", at "94% GPU utilisation"** | Retracted 2026-08-16. | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **the CUDA replacement, "0.8 s of 861, or 1.0009x"** | Do not quote this either, and it is the more instructive of the two. | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **any A/B of a numerical knob that rests on one rendered clip per arm, including...** | Not controlled evidence about the knob. | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **"the DiT is sensitive to the seven H3 marker rows", and the 0.135 relative-L2...** | Withdrawn 2026-08-27, the same day, by the control it never had. | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **the 2026-08-15 ordering arms read as a matched pair** | They were never paired. | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **the fl2va and ref2va time curves differ by "7.8% relative", as stated in...** | UNSUPPORTED, marked 2026-09-03. | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **block 49 "carries about four times any other block's error with every block...** | Withdrawn 2026-09-03, the same day, on Codex's review. (no `dense_blocks` list touches it) | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **the Base16 records' tau_1.0 arm is "the shipped tau", and (briefly) "an upper...** | Withdrawn 2026-09-03. (an upper bound on the shipped call) | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **any reading of the 2026-09-04 ladder verdict as settled** | Not withdrawn | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **"PDD8 is worse than sage at sixteen steps" or "Sol on two of eight steps is...** | Bounded, 2026-09-05. | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **"the bake fixed PDD8" or "the bake changed nothing", read from the bake pair...** | Bounded, 2026-09-05. | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **"PDD's audio loss is ComfyUI's audio change of variable", and any reading of...** | Refuted analytically, 2026-09-25. | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **"Sol without sage is worse than the sage floor" as settled, read from the...** | Bounded, 2026-09-05. | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **"the encoder pathway carries identity on its own" or "reference rows are just a...** | Not withdrawn | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |
| **Sol and the shipped sage fallback were compared "on one footing" via...** | Withdrawn 2026-09-03. | [detail](wiki/archive/2026-09-05_retracted_claims_part2.md) |

**362 is the max length, and no shorter count is a limit. Owner decision,
2026-08-16; made the only cap on 2026-10-15.** The reference pipeline
(diffusers) stops one grid step earlier, because its `max_duration` is a
hard-coded 15.0 s and 362 is 15.083 s: a fact about diffusers, answered by
`h3_rules.reference_would_emit()`, never a fact about the checkpoint. This
repo once turned it into "362 is illegal" and withdrew a bench run over it.
`h3_rules.MAX_LENGTH` is the ceiling and `workflows/h3_config.py::LONG_LENGTH`
is what shipped graphs render; today they are equal. A chained window's
default is the longest length whole on both clocks (`h3_rules.chain_lengths`;
`docs/h3_resolutions.md`, "Whole on the audio clock"), derived, never typed.
Longest is not best: attention is quadratic in the sequence, so a window
takes the frames its shot needs.

**What 362 rests on, stated because this table exists for exactly this.** One
upstream statement recorded 2026-08-14 (`6e85e48`) with no artifact attached,
plus LightX2V shipping a 362-frame config. MiniMax's README gives a rounded
"4-15 seconds" and the official checkpoint configs state no frame limit at all,
so the primary source neither confirms nor refutes it. This is a decision taken
on thin evidence, not a measurement — do not quote it as one.

---

## These hold

Kept short on purpose — every row is something a second reader confirmed or an
instrument that has been shown red.

### Settled about H3, moved here from `CLAUDE.md` on 2026-09-03

`CLAUDE.md` keeps one line per fact; this is the ledger row. The full argument
for each, as it stood when it was written, is in `docs/rules_history.md`.

- **The released text encoder is byte-identical to stock
  Qwen3-VL-32B-Instruct**, every shard, by hub LFS SHA-256.
  `data/results/2026-08-25_released_encoder_is_stock.json`. MiniMax shipped
  no encoder post-training, so every row of its embedding table is stock.
- **The seven H3 marker rows sit with the untrained padding tail** in the
  encoder's input embedding table,
  `data/results/2026-08-21_h3_token_embeddings.json`. A consequence of the
  row above, not a finding about markers: single-id-against-frozen-encoder and
  BPE'd-so-never-seen both predict it. **Corrected 2026-09-03, hours after
  this row was written**: it said the discriminator, `release_id` against
  `mean_init_rows`, "has not been run", copied from the `CLAUDE.md` sentence
  the 2026-08-31 postmortem had already flagged. The prediction-delta form DID
  run (`data/results/2026-08-27_marker_epsilon.json`) and was refuted by its
  own control (`data/results/2026-08-27_marker_epsilon_control.json`: the
  markers sit at the low end of the control range); the do-not-rely row above
  owns that. What has not run is the blind-render form. A session read the
  untrained rows as "the fixed tokenizer is wrong" on 2026-08-27 and
  retracted.
- **The marker ids are fixed by loading the release's tokenizer directory**
  and appear in no JSON literal; the row at the top of this section owns it.
- **The DiT was trained on MiniMax's own prompt structure, and it differs by
  mode.** `vendor_guides/`; `docs/prompting.md` is the working manual.
- **1344x768 is a trained canvas, and a cheap canvas has inverted a finding.**
  `docs/h3_resolutions.md`.
- **PDD quality is governed by the sigma schedule's coarseness and where it
  lands, not by the evaluation count.** Two arms with the same count and the
  same block widths in a different order differ materially; under shift 12
  the tail is what matters, and the uniform four-evaluation partition is the
  only legal four-block partition, so its coarse tail is forced. Both streams
  degrade together (raw video rel L2 hides it behind the DC term). Every
  coarseness statistic ranks the arms identically, so no partition experiment
  can attribute the effect to audio. *2026-09-25: and there is no audio-specific
  transform error to find. Core's carry is exact under Euler, so the audio loss is
  PDD's own. The clause that used to close this bullet, "vary the transform at fixed
  partition", is withdrawn (`research/pdd/2026-09-25_upstream_pdd_comparison.md`).*
  `docs/research/pdd/audio_under_pdd.md`.
- **The encoder this install loads is `h3_config.MODELS["clip"]`.** The v2
  AWQ lane was closed on 2026-08-27 rather than adopted, and its code was
  deleted on 2026-09-13 (`docs/wiki/decisions.md`); the loader every graph
  wires is `MiniMaxH3EncoderLoader`. A different question from the first row:
  that one is about the released weights, this one about which local artifact
  is wired. **Closed is not refuted** (owner, 2026-09-20): the artifacts were
  badly executed, we controlled the calibration and every other knob, and
  priorities shifted -- so the holdout is not evidence that quantising our own
  encoder is unpromising. All four of its arms hold the vision tower at BF16,
  which the record does not say, so it is silent on tower precision.
  `docs/wiki/decisions.md`.
- **Reference sizing: `docs/h3_references.md` is the authority.** `size_policy=max`
  with the vendor short edge matches the vendor; `dit_short_edge` targets the
  shorter side and only shrinks unless `allow_upscale`; both live on
  `MiniMaxH3AppendRefImage` and are read only under `max`.
  `MiniMaxH3ReferenceFit` was deprecated on 2026-08-28 and deleted in 0.173.0. `qwen_view` on the append node can give the encoder a view of its
  own (`separate`, pre-filled from `h3_rules.REF_QWEN_SHORT_EDGE`); a flat
  `qwen_short_edge` whose 0 meant shared until 2026-08-31. **Since 2026-09-13
  the node's defaults are vendor parity**: `max`, a 2048 short edge,
  `allow_upscale` on, `qwen_view=shared`, one prepared still for both the
  video VAE and Qwen3-VL, as sglang, diffusers and DiffSynth do
  (`docs/research/sglang_h3_pipeline.md` "Reference stills"); read them from
  the node's `define_schema`. The separate 512 view that shipped from
  2026-08-27 rested on one render at one seed (CHANGELOG 0.82.0).
  **Since 2026-10-03 `qwen_view` is `separate` at 512 again, by the owner's
  decision**, on the node, the constant and the generated graphs: three clips
  of the finish graph at one seed passed their eye and ear and sampled faster
  (`bench/results/2026-10-03_sol_output_check.md`, section 5). Not blind, one
  seed, not the vendor's path, and on the one dialogue scene the mix is
  louder at this view; `docs/wiki/decisions.md` has the evidence and what it
  does not cover. The other three defaults stay at parity, and the instruments stay
  on `shared` (`workflows/h3_config.py::REF_QWEN_SHORT_EDGE`).
  `bench/refview2_arms.json` is the ablation built to settle `qwen_view` and
  `allow_upscale`, rendered on 2026-09-13 and never scored. `MiniMaxH3ReferenceConditioning`'s preview prices both
  copies of every reference before anything is encoded (the `MiniMaxH3ReferenceReport`
  node that drew them was deleted in 0.174.0).

- **The marker ids are the release's own, fixed by construction rather than by
  literal.** No file in the release assigns `<d>`..`<|caption_end|>` an id --
  checked across all 27 tokenizer/vocab files -- which has now generated the
  same false finding twice ("core derived them and may have guessed"). Loading
  the release's own tokenizer directory with the standard loader assigns
  **151669-151675** and tokenizes all seven as single tokens, `len(tokenizer)`
  151676 against a 151643 base plus 26 added. Core's "ids fixed by the released
  tokenizer" is correct. Recorded in
  [`2026-08-27_marker_tokenization_alignment.json`](../data/results/2026-08-27_marker_tokenization_alignment.json)
  so the third reader finds the answer instead of re-raising it.
- **Two of the seven markers reach anything we render or calibrate.** `<d>` and
  `</d>` appear in shipped graphs, in the v2 calibration bundle and in all three
  holdouts; `<|cutoff|>`, `<|lyrics_start|>`, `<|lyrics_end|>`,
  `<|caption_start|>` and `<|caption_end|>` appear in **zero** graphs and zero
  rows of the pool the selection drew from. So every encoder measurement here --
  Gate 5, the overfit test, the four-encoder table -- covers two of seven.
  `bench/marker_corpus/` is the only surface that reaches all of them.
- **The prefix length is a coordinate origin, not just a length.**
  `comfy/ldm/minimax/model.py::PackedLayout.__init__` sets
  `cursor = float(text_len)` and hands it to `_audio_grid` and `_video_grid` as
  their temporal origin, so any arm that changes token count moves the whole
  target timeline in RoPE t-space. The prefix-end to target-start gap is
  invariant; what moves is the relative distance from early prefix rows,
  graded to zero at the boundary. An arm that changes tokenization is therefore
  never prefix-local, and a "few tokens in a large sequence" argument about one
  is measuring the wrong thing.
- **The CUDA seam works.** Live render, `cuda-int8` in the log, override
  chained, 50 forwards composed. Not a source read.
- **One `optimized_attention` call in source**, `comfy/ldm/minimax/model.py`,
  reached by **52 modules** — 50 DiT blocks plus 2 token-refiner blocks, all
  sharing one `Attention.forward` (`num_layers=50`,
  `token_refiner_num_layers=2`). "One call site" is not "one call", and the
  refiner blocks see only the text span while the 50 see the full sequence.
- **Sage runs 5 of 16 steps under Sol** at the shipped window — verified at two
  layers of the node source and cross-checked against this repo's own
  sigma-window table, which independently gives 11/16 sparse.
- **`reuse_qkv_memory` is numerically identical** to the normal entry, six
  digits. It cannot change output. What it *buys* is unmeasured.
- **Both backends are arithmetically equivalent** at `T=512` fidelity.
- **All 34 API graphs carry `LONG_LENGTH`**, read from their widgets.
- **Every `bench/check_*.py` passes**, and the ones added today were each shown
  red first: `check_sol_kernel` (4 cases), `check_bench_matches_shipped`,
  `vendored`, `node_version`.
- **`int8_convrot` sits about three times closer to the bf16 release than
  `fp8_scaled`.** Measured 2026-08-21 against the release itself
  (`bench/analyze_quant_delta.py`; records
  [`data/results/2026-08-21_quant_delta_fl2va.json`](../data/results/2026-08-21_quant_delta_fl2va.json)
  and its ref2va twin): median relative delta per block linear 0.0091 against
  0.0265, on both checkpoints, cosine 0.9999 against 0.9996. fp8's error is
  flat to four digits across all 200 modules while int8's varies by module kind
  and is worst on `attn.out_proj`. **Both files sit at their format's floor**:
  quantizing one release weight in the script itself, scalar-scaled e4m3
  against per-row int8 through the same rotation, reproduces 0.02651 and
  0.00882 where the shipped files read 0.02651 and 0.00882. The gap between
  the formats is the formats, not the calibration. Every tensor neither file quantizes is
  byte-identical between them, so nothing else is in the comparison.
  **Stored weights only**: fp8 carries an `input_scale` on 150 of its 200
  quantized tensors, int8 stores no activation scale at all, and what either
  does with activations at run time is invisible to a weight comparison. The controlled runtime version
  is `docs/open_experiments.md` #22's fixed-input first-step forward.
- **The bf16 release and the Comfy repack order `attn.qkv_proj` rows
  differently.** The release interleaves per head, `[head][q|k|v][head_dim]`;
  the repack concatenates, `[q|k|v][head][head_dim]`, which is what
  `comfy/ldm/minimax/model.py` splits on. **A "correction" to this bullet on
  2026-08-29 was itself wrong and is withdrawn**: it claimed the release ships
  no fused qkv and renamed the interleaved order a "third-party" convention.
  That came from reading only `coderef/MiniMax-H3/transformer/`, the
  **diffusers-format** copy (`MiniMaxH3Transformer3DModel`, 638 keys, split
  `to_q`/`to_k`/`to_v`). The release also ships its **native** format at
  `FL2VA/transformer/` and `Ref2VA/transformer/` (`MiniMaxH3DiTModel`, 535
  keys), and that one carries `blocks.N.attn.qkv_proj.weight` — 52 of them —
  under key names identical to ComfyUI's. This bullet is about that format and
  it was right as written. Compared as stored, a qkv weight
  reads a relative delta of 1.40 against its own origin with cosine ~0, in
  both quantizations equally; reordered, it reads the same ~0.9% as every
  other int8 linear. `analyze_quant_delta.py` measures both readings and
  refuses to run if the reordering is not the better one.
- **`DeepBeepMeep/MiniMax-H3`'s FL2VA files against `Comfy-Org/MiniMax-H3`'s,
  measured 2026-08-25** by `bench/compare_dit_checkpoints.py` on headers and
  range-fetched samples (records `bench/results/2026-08-25_dit_fl2va_*.json`).
  The int8_convrot files, pruned and unpruned, are Comfy's bytes under a
  different header. The bf16 files are Comfy's bytes except `qkv_proj`, which
  DBM stores in the **release order** above; head 0's k rows sit where
  ComfyUI's split expects head 1's q, so a DBM bf16 file is a WanGP file and
  not ComfyUI-loadable as-is. DBM `pruned_rank8` and Comfy `pruned` are the
  same rank-8 AdaLN factorisation in a sign-flipped basis (basis-change
  residual 3e-7), with the modulation agreeing to 2e-4 relative, which is
  Comfy storing the factors in F16 where DBM keeps F32. DBM `pruned` with no
  suffix is a **rank-64** factorisation on a 1,001-point grid that Comfy does
  not publish; against rank-8 its modulation differs by 1e-5 to 7e-5 across t.
- **The `adaln_all` hybrid applies ref2va's AdaLN linears to fl2va's curve
  table, measured 2026-08-25** with the same tool. `bench/build_hybrid.py`
  copies `blocks.*.adaln_proj.linear.*` and leaves the shared `adaln_t_table`
  on fl2va; the two partitions' tables span nearly the same subspace (residual
  5e-5) but are not the same table, so the hybrid's modulation sits 0.1-0.2%
  from ref2va's own at every block, an order above the rank-8 truncation and
  an order below the 2-4% by which ref2va's AdaLN differs from fl2va's. A
  partial hybrid cannot avoid this (one table serves both parents' blocks); the
  all-blocks build could copy ref2va's table and has not been rebuilt.
- **The H3 Turbo LoRAs across `lightx2v/Minimax-h3-Turbo` and
  `DeepBeepMeep/MiniMax-H3/loras`, measured 2026-08-25** by
  `bench/compare_lora_files.py` (records `bench/results/2026-08-25_lora_*.json`;
  whole sampled tensors per module kind, every alpha scalar read). DBM's five
  `lightx2v_*` files are lightx2v's diffusers files byte for byte plus one
  `.alpha` scalar per module, equal to lightx2v's metadata alpha where one
  exists (128 for v1.0 and v1.1, 8 for 8-step and ref2v). **v0.1 declares no
  alpha anywhere and DBM assigned 16**; its origin is in neither file, and
  against rank 128 it is an 8x smaller scale than v1.0's. lightx2v's
  `comfyui_bf16` variants are exact conversions: q, k, v fused block-diagonally
  into one rank-384 `qkv_proj` LoRA (each band byte-identical to its source),
  `fc1` lora_B halves swapped as the file's `swi_glu_mapping` declares,
  everything else byte-identical, fused alpha scaled with the rank so the
  effective scale is unchanged. The shipped `minimax_h3_turbo_v4_step600_ema`
  is DBM's `larryvrh_v4_step600_ema`, identical on every sampled tensor; DBM's
  `turbo_4step_ema_ckpt850` shares its modules and metadata and differs on
  every tensor, so it is another checkpoint of that recipe, not a re-save.
- **Withdrawn 2026-09-03: "the node is vendored, symlinked and hash-pinned, so
  the file ComfyUI loads cannot drift from the tracked one."** That described
  the vendored `SolAttnMiniMax`. Since 2026-08-30 `vendor/sol_attn_minimax.py`
  is a read-only reference that is not loaded, and the node the graphs wire is
  `MiniMaxH3SolAttn` in `sol_attn_h3.py`; `bench/check_sol_node_equivalence.py`
  is the guard that relates the two. *2026-09-27: that node is replaced by
  `MiniMaxH3Sol` (`docs/research/2026-09-27_sol_node_redesign.md`).*

---

## Enumerated consumers, checked mechanically

A retraction is done when every **consumer** of the claim is enumerated, not
when the claim is corrected. That is today's most-repeated failure in one
sentence: "sage gets nothing" was retracted and went on producing wrong answers
in three more places, and the 23-point swing was retracted in this file while
still carrying an argument in `docs/bench_plan.md`.

The block below makes that enumeration executable.
`bench/check_retraction_consumers.py` fails when a retracted phrase appears in a
file **not** listed for it. There is no caveat-detection and no judgement: a
mention inside a `RETRACTED:` block and a live use are indistinguishable to a
matcher, so the check does not try. It answers one decidable question — has this
claim reached a file nobody signed off on — which is the failure mode that
actually occurred, four times, all in files that acquired the claim *after* the
retraction.

Adding a file here is a claim that someone read that occurrence and it is
correct in context. Do not add one to silence the check.

```retraction-consumers
ONE LINE PER FIELD. The parser reads only the line beginning `ALLOW:`, so a
wrapped continuation is silently dropped and the row enumerates fewer files
than it appears to. That fails closed -- the missing files show up as
unlisted consumers -- but it wastes a run, which is how it was found on
2026-08-16.

`ALLOW: (none)` means the phrase must appear in NO file: the claim was deleted
repo-wide rather than caveated, so it has no consumers and any occurrence is a
reintroduction. Spelled out rather than left empty, because a bare `ALLOW:`
is treated as a truncated line and fails the parse.

PHRASE: 2.7x more accurate
ALLOW: docs/evidence.md
WHY: the figure was DELETED repo-wide on 2026-08-16, not caveated, so the
     allowlist collapses to this ledger. The phrase is retained here so the
     check keeps a tripwire on it: if it reappears in any file, someone has
     reintroduced a withdrawn number. Still deliberately NOT matching bare
     "2.7x" -- attention.py and README.md use that string for a speed figure
     against torch flash, a different and correct claim.

PHRASE: 23-point swing
ALLOW: docs/evidence.md docs/bench_plan.md docs/SOLATTN.md
WHY: bench_plan's is inside the retraction block that replaced the argument;
     SOLATTN's are its "Do not rely on" table and the reference section.

PHRASE: sage gets nothing
ALLOW: docs/evidence.md docs/SOLATTN.md
WHY: both are retraction statements naming it as withdrawn.

PHRASE: zero DiT calls
ALLOW: docs/bench_plan.md docs/checks.md docs/checks/index_02.md
WHY: bench_plan's is inside a "RETRACTED 2026-08-14" bullet stating the
     5-of-16 correction. checks.md quotes it as the worked example of why this
     check is an allowlist rather than caveat-detection -- and it was caught
     by the check on the commit that added it, which is the behaviour wanted.

PHRASE: exactly one attention site
ALLOW: docs/SOLATTN.md
WHY: the sentence that contains it says the page claimed this until 2026-08-14
     and corrects it to 52 modules.

PHRASE: 38 text rows
ALLOW: docs/SOLATTN.md
WHY: both name it as a smoke-harness figure rather than a shipped-graph one.

PHRASE: worth 1.16x alone
ALLOW: docs/evidence.md docs/morton.md docs/SOLATTN.md docs/open_experiments.md docs/bench_plan.md CHANGELOG.md
WHY: Triton, 362 frames, stacked on int8, retracted 2026-08-16 when a dense
     control showed the CUDA permutation free. morton.md owns the
     retraction; SOLATTN.md's is the "do not rely on" row and the corrected
     Configuration-findings bullet; open_experiments.md and bench_plan.md
     name it as withdrawn in the paragraphs that used to argue from it;
     CHANGELOG.md records the retraction as history and is not rewritten.
     Enumerated the same day the retraction was made -- the earlier three
     same-session retractions were not, and one of them (this one) went on
     being asserted in SOLATTN.md's Configuration findings for hours
     afterwards. That is the open question raised in
     internal/postmortems/2026-08-15_session_morton-sampler-and-refs.md,
     answered by the failure recurring rather than by argument.

PHRASE: essentially rewritten
ALLOW: docs/evidence.md docs/h3_ref2v_distillation.md CHANGELOG.md
WHY: the 1.92 `final_layer.adaln_proj` figure, withdrawn 2026-08-20 as a
     curve-form basis artifact. h3_ref2v_distillation.md keeps it inside the
     parenthesis that withdraws it; CHANGELOG.md records it as history.

PHRASE: replaced rather than adjusted
ALLOW: (none)
WHY: the same figure in its other spelling, carried by h3_config.py and by
     bench/analyze_ref_lora.py until 2026-08-20; h3_config.py now says so
     without the phrase, and analyze_ref_lora.py was deleted 2026-08-21 with
     the rest of the reference LoRA. No consumer is left, so this row is a
     tripwire rather than an enumeration: it went `(none)` on 2026-08-25,
     having warned as a stale allowlist since the cleanup.

PHRASE: 94% GPU utilisation
ALLOW: docs/evidence.md docs/morton.md docs/SOLATTN.md docs/open_experiments.md CHANGELOG.md
WHY: the same measurement and the same retraction. Listed separately because
     the two halves of that claim were quoted independently -- h3_config.py
     carried the utilisation figure as the mechanism story for the 1.16x, so
     either half can reappear without the other.
PHRASE: no `dense_blocks` list touches it
ALLOW: docs/evidence.md CHANGELOG.md
WHY: withdrawn 2026-09-03; CHANGELOG 0.99.27 quotes it inside its own
     retraction, the research record restates the claim in other words.

PHRASE: an upper bound on the shipped call
ALLOW: docs/evidence.md
WHY: the label lived an hour on 2026-09-03 and was replaced everywhere by
     "unsunk diagnostic"; any other occurrence is a reintroduction.

PHRASE: one footing
ALLOW: docs/evidence.md CHANGELOG.md bench/measure_sol_exact_variants.py docs/research/2026-09-03_sol_exact_pquant_and_base_capture.md
WHY: withdrawn 2026-09-03; the grader's docstring and CHANGELOG 0.99.27
     quote it to say what was wrong with it. A new mention must say which
     rows and which reference before it means anything. The research
     record's closing section (read 2026-09-05) says both: every retained
     Base16 cell, against the offline exact_all_routed numbers, which the
     replay reproduced; that is the mention the rule permits.
```

**What this does not defend, and the gap is not small.** The sharpest decay
found on 2026-08-14 has no string to match: `docs/bench_plan.md` read "one
**345-frame** video reference", where 345 is the *reference* length and looks
like the shipped config, concealing that the *target* was 362. The retracted
thing was the **pairing** of two numbers, not any token in the sentence. No
phrase matcher can see that, and a check that appears to cover a class it
silently omits is how the 2026-08-13 validator bug worked — so it is said here
rather than left to be assumed.

**The general form, named 2026-08-16 after it happened again.** This check
defends the *spelling* of a retracted claim, not the claim. A retracted
**measurement** can reappear in any wording containing no listed phrase, and
**derived figures are exactly that shape** — a spread between two withdrawn
numbers, a ratio of them, a rescale, a rounding. The instance: after the
accuracy ratios were withdrawn, `CLAUDE.md` kept a sentence saying all three
fp8 variants land within a stated tolerance of each other — a mean_rtol spread
from the same withdrawn sweep — nine lines below the paragraph announcing there
was no numeric half left. The commit that removed its neighbours left it, and
its author read that paragraph twice the same day. Two properties made it
invisible: it carried no retracted token, and it read as a conclusion rather
than as a number.

So when withdrawing a measurement, grep for the *quantities* it produced and
not only for the sentence it was written in.

**And it cannot be closed by adding a row, which was tried on 2026-08-16 and
reverted the same hour.** The ledger is an allowlist: every phrase needs at
least one file that legitimately contains it, and the `stale_allowlist` case
warns when a listed phrase is absent from its own allowed files. A spelling
that should appear **nowhere** therefore cannot be expressed — listing it emits
a permanent warning on every run, which is the crying-wolf failure this repo
treats as worse than no check. The model fits a claim with a legitimate home
and a forbidden everywhere-else; it does not fit a claim that is simply banned.
Catching those needs a different instrument, and until one exists this
paragraph is the instrument.

---

## Environment, because it is not in git

The historical August 2026 notes on `comfy-kitchen` Sol-Attn wheel tags (`0.2.31+sol.dae00a1`), AdaLN rank-8 SVD pruning residuals (`0.1-0.2%` modulation, `5.6-9.4%` first-step velocity L2 vs `12-21%` fp8 quantization delta), and host GPU board power limits (`docs/hardware.md`, `bench/hwinfo.py`) are archived verbatim in [`docs/wiki/archive/2026-09-05_retracted_claims_part2.md`](wiki/archive/2026-09-05_retracted_claims_part2.md#environment-because-it-is-not-in-git). For current live environment setup, see [`docs/comfy_notes.md`](comfy_notes.md).
