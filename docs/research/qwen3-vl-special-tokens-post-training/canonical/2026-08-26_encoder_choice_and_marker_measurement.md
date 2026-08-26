# What the AWQ lane could never have moved, what an encoder costs to hold, and how the marker question gets answered without a trainer

**Status:** Authoritative facts; the two recommendations at the end are open for
the owner
**Recorded:** 2026-08-26
**Reading it against:** [`2026-08-25_v2_launch_record.md`](2026-08-25_v2_launch_record.md)
owns the Gate 5 verdict and the four-encoder table and is not restated here.
This record answers three questions that came after it and were not in it.

## 1. The v2 calibration could not have moved either thing the lane is named for

The lane is `qwen3-vl-special-tokens-post-training`. The v2 run was a W4A16 AWQ
calibration. Those are not the same object, and the gap is structural rather
than a matter of the metric Gate 5 chose.

**SOURCE.** The quantization scheme targets `["Linear"]`.
[`bench/h3_awq_recipe.py`](../../../../bench/h3_awq_recipe.py) says why the
embedding needs no ignore entry -- it is an `nn.Embedding`, never a target --
and its boundary assertion checks that the embedding and the output head carry
no scheme rather than trusting the pattern list. The vision tower, patch merger
and DeepStack mergers are excluded the same way, recovered by module class.

**MEASURED.** The assertion is not what settles it; the holdout captures are.
In [`2026-08-25_v2_holdout_layer50.json`](../../../../bench/results/2026-08-25_v2_holdout_layer50.json)
the `layer0_input` relative L2 is exactly **0.0** on every row, both geometries,
both W4 arms. The token embedding table and the entire vision path are
byte-identical across BF16, v1 and v2. Whatever v2 was, it was not a different
vision encoder and it did not touch the seven H3 marker rows.

**INFERENCE, from those two.** The only thing that differed between v1 and v2
was the 4-bit approximation of the decoder linears. Calibration data can shift
where that error lands; it cannot teach the model anything, and it cannot reach
the untrained marker rows or the vision tower at all. Grading it against BF16
was correct for the object that was built -- a quantization's ceiling is the
full-precision model -- and irrelevant to the objective in the lane's name. Both
hold at once.

**MEASURED, and it closes the one lever that did exist.** Calibration data
could in principle bias the scale search toward H3's schema positions. The pool
cannot supply them: in
[`2026-08-25_v3_selection_max_no_upscale.json`](../../../../bench/results/2026-08-25_v3_selection_max_no_upscale.json),
`<d>` and `</d>` occur in 505 of 1,028 pool rows and the other five H3 markers
in **zero**, while the whole label scaffold -- `<Picture i>:`, `<Video k>:`,
timestamps -- is 952 of 265,922 tokens. A scale search cannot be biased toward a
schema that is a third of a percent of the token mass and missing five of its
seven markers.

**Consequence.** Do not propose calibration data as a route to marker,
prompt-structure or vision-encoding alignment. `active_plan.md` already scoped
the v2 run narrowly ("do not... begin special-token training as part of this
work"); what nobody wrote down before the run is that no calibration population
could have reached those questions in principle. That is the sentence this
section exists to make available.

## 2. What each encoder costs to hold, by component

**MEASURED, 2026-08-26.** Producer
[`bench/measure_encoder_footprint.py`](../../../../bench/measure_encoder_footprint.py),
record [`2026-08-26_encoder_footprints.json`](../../../../bench/results/2026-08-26_encoder_footprints.json).
Safetensors headers only; no tensor data read. The **H3 path** is decoder layers
0--49 plus the embedding table and the vision tower, with later layers and the
output head dropped as `comfy/text_encoders/minimax.py` drops them on load. It
is weights only: activations, offload staging and allocator overhead are not in
it.

| encoder | on disk | H3 path | decoder | embeddings | vision tower |
|---|---:|---:|---:|---:|---:|
| BF16 release | 62.13 | 47.97 | 45.41 | 1.45 (BF16) | 1.11 (BF16) |
| INT8 ConvRot | 25.28 | 25.28 | 22.72 (I8) | 1.45 (BF16) | 1.11 (BF16) |
| NVFP4 AWQ | 14.61 | 14.61 | 12.78 | 0.73 (I8) | 1.11 (BF16) |
| W4A16 AWQ v1 | 18.99 | 14.27 | 11.71 | 1.45 (BF16) | 1.11 (BF16) |
| W4A16 AWQ v2 | 19.00 | 14.27 | 11.71 | 1.45 (BF16) | 1.11 (BF16) |

GiB throughout. Read beside the fidelity column of the four-encoder table,
which the launch record owns.

Three things this table says that file size did not:

- **The encoder of record does not fit a 24 GB card.** INT8 ConvRot's H3 path is
  the whole file: it already carries only layers 0--49, so there is no depth
  slack to find.
- **The vision tower is 1.11 GiB, under five percent of every artifact.** There
  is nothing to win by compressing it and it is the part the lane cares about.
- **The embedding table is BF16 in three of the four artifacts.** NVFP4 already
  ships it at I8 for 0.73 GiB, so the format is proven in a file already on the
  box.

**MEASURED, and it bounds what fitting is worth.** The occupancy instrument on a
real arm-B render
([`2026-08-25_refview_b_qwen2048_int8_occupancy.json`](../../../../bench/results/2026-08-25_refview_b_qwen2048_int8_occupancy.json))
put the INT8 encode at 59.6 s of a 691 s render, peaking at 20,470 MiB with
95.8% mean GPU utilization and 22.3% mean memory-interface. ComfyUI streams the
oversized file layer by layer and the per-layer compute hides the transfer, so
the encode is compute-bound, not streaming-bound. A perfect fit therefore wins
at most that 59.6 s.

## 3. Recommendation: no further own-calibration quants for this card

**RECOMMENDATION, owner's call open.** For a 24 GB target, stop building AWQ
candidates. The reading:

- The 4-bit grid is the floor and the launch record measured it: both W4
  artifacts sit at 0.33--0.38 median relative L2 on every population and
  calibration data moves that by about a tenth either way.
- NVFP4 already dominates W4A16 at the policy that ships. `active_plan.md` makes
  the 2048 short edge the primary reference-still policy; NVFP4 reads 0.147
  there against W4's 0.312, at 14.61 GiB against 14.27. It loses marginally at
  `max_no_upscale`. If a small encoder is wanted, that file exists.
- There is nothing to calibrate on the INT8 path. ConvRot stores `W @ H^T` in a
  **constructed** Hadamard basis (`_build_hadamard` in
  [`bench/analyze_quant_delta.py`](../../../../bench/analyze_quant_delta.py)) --
  a data-free transform. No calibration set can improve or harm it.

What to do instead, in the order I would do it:

1. **Cache the conditioning.** If the working loop is seed iteration on a fixed
   prompt and reference set, the encode happens once rather than per render and
   the whole VRAM question stops mattering for that workflow. Design note:
   [`conditioning_cache.md`](../../conditioning_cache.md).
2. **Mixed precision chosen by measured per-module sensitivity.** The format
   supports it already -- group size is read from each tensor's own
   `comfy_quant` marker, so one file can carry I8 on some linears and 4-bit on
   others. Shaving INT8's H3 path to fit needs about 2.8 GiB, of which the
   embedding table at I8 is 0.72.
3. **Leave the vision tower alone**, per the table above.

**The measurement that is missing, and it is cheap.** No encoder-side
per-module weight-error record was found under `bench/results/` on 2026-08-26.
`analyze_quant_delta.py`'s per-module work is on the **DiT**
(`blocks.N.attn.qkv_proj`, `mlp.fc1/fc2`), and the four-encoder table is
whole-encoder layer-50 only. Per-module sensitivity for Qwen's decoder is
weights-only -- no renders, no calibration, no GPU time -- and it is what turns
step 2 from a guess into a knapsack: minimise predicted layer-50 error under a
byte budget. Anyone picking that up should also check whether the widest MLP
input (`down_proj`, the same tensor family that dominated AWQ's host cache)
tolerates 4 bits, because it is where the bytes are.

## 4. Recommendation: no encoder fine-tune or LoRA before the marker arms run

**RECOMMENDATION, owner's call open.** Two of the three things usually named
together here are not weights problems at all:

- **2048 short edge** is a serving policy. The vision tower patches arbitrary
  resolution; no weight encodes a short-edge target. Whether the policy helps is
  the Gate 6 blind-pair question on the three `h3_probe_refview_*` graphs, and
  that has not run.
- **Prompt structure** already matches the vendor.
  [`h3_references.md`](../../../h3_references.md) grades ComfyUI's presentation
  against sglang's stage: same string, same place, no chat template.

**Special tokens are the only genuine weights question, and the frozen DiT is
the argument against training them.** **MEASURED**
([`2026-08-25_released_encoder_is_stock.json`](../../../../bench/results/2026-08-25_released_encoder_is_stock.json)):
the released `text_encoder/` is byte-identical to stock Qwen3-VL-32B-Instruct
across all 14 shards. So the release pairs the DiT with an encoder whose seven
marker rows are untrained. **INFERENCE:** training those rows moves the encoder
off the distribution the DiT was paired with -- drift, not alignment.

**UNKNOWN, and it must stay labelled that way: we do not know the DiT was
trained on stock Qwen3-VL.** The measurement above is about the *released*
encoder. What MiniMax used during DiT training is not observable from the
release, and [`baseline.md`](baseline.md) records it as unknown.

### How the question gets answered with no trainer

Not by teaching -- by **selecting among representations that already exist** and
measuring the DiT's response. Four arms, declared in
[`bench/marker_corpus/compiled.json`](../../../../bench/marker_corpus/compiled.json)
with the runtime in [`marker_arms.py`](../../../../marker_arms.py):

| arm | what changes | trained rows? |
|---|---|---|
| `release_id` | nothing; the seven dedicated IDs | no, untrained stock tail rows |
| `legacy_bpe` | tokenizer: marker text as ordinary BPE, via an unpatched tokenizer **selected by vocabulary, not by constructor name** | yes |
| `stripped` | prompt text: marker strings removed, every other character kept | n/a |
| `mean_init_rows` | the seven rows replaced by the embedding-table mean, as an offset-keyed `set` patch on a clone -- never the shared module, never a file | n/a |

`legacy_bpe` is the only arm carrying trained representations for the markers.
So the question is not which arm is better in the abstract but **which
distribution the frozen DiT learned to read**.

Three instruments, all built:

- **Encoder level, deterministic, no renders.** Capture layer-50 per arm and
  compare on aligned positions. Done once already
  ([`2026-08-21_h3_marker_token_states.json`](../../../../bench/results/2026-08-21_h3_marker_token_states.json)):
  marker representation moves the encoding by relative L2 0.034 on a short voice
  line to 0.16 on a dialogue-heavy prompt. Against the four-encoder table's
  0.021--0.027 for INT8 and 0.31--0.37 for W4, marker representation is a larger
  perturbation than choosing INT8 over BF16, and free.
- **DiT level.** The encoder result proves the arms differ, not that the DiT
  cares. That needs renders as a distribution, never a pair:
  `run_graph_arms.py` -> `blind_batch.py` -> the scoring app ->
  `score_session.py`, under `eval_comparison.md` section 3.
- **Structure level**, for "does this piece of prompt structure reach the
  model", with no perceptual judgement:
  [`bench/measure_dit_prefix_attention.py`](../../../../bench/measure_dit_prefix_attention.py).
  Its ref2va version needs the capture path to dump token tags and vision spans
  first.

### Why the arms are also the discriminator for section 4's UNKNOWN

The seven rows are untrained but not zero -- they are some initialisation
vector. **INFERENCE**, stated before the arms run so it cannot be fitted to
them:

- `release_id` reliably beating `mean_init_rows` means the DiT learned to read
  those *specific* untrained vectors, which is only possible if it trained
  against an encoder carrying exactly those rows. Evidence that the released
  encoder is the training encoder.
- The two being indistinguishable means the dedicated IDs carry no learned
  meaning to the DiT, and the special-token lane closes cheaply.
- `legacy_bpe` winning means training used the pre-patch spelling, and the fix
  is a tokenizer choice rather than anything in the weights.

**Readiness caveat.** The corpus is currently five scenes and says so of itself
-- "a seed set, not the frozen evaluation corpus; freezing is the owner's act"
-- and declares two missing cells (standalone reference audio with a referenced
speaker; reference video with a soundtrack). The arms can run on what exists; a
quotable claim wants the coverage the brief names.

## What this record does not establish

- Anything perceptual. Every recommendation here rests on weights and encoder
  states. Gate 6 has not run.
- That mixed precision recovers INT8's fidelity. Section 3 step 2 is a proposal
  with a named missing measurement, not a result.
- What the DiT was trained against. Section 4 says how that becomes answerable,
  and answers it nowhere.
- Any change to the deployed artifact, the encoder of record, or the card order
  in the 2026-08-26 point handoff. Nothing here supersedes an owner decision.
