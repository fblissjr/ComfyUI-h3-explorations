# One render, stage by stage: code, owner, guard, reference

The cross-index. For each stage of a render: **our code**, the **document that
owns** it, the **check that would go red** if it broke, and the
**implementation to compare against** when you need to know what it should do.

Written by hand; no generator touches it.

**"nothing" in the guard column is the useful entry.** It is not a to-do list;
the root `AGENTS.md` rule is that a check is added only when something escapes
the existing ones. It is there so nobody mistakes an unguarded
stage for a guarded one. The standing audit is
[`docs/method/checks.md`](../method/checks.md)'s uncontrolled-requirements table, which is the
authority; this column routes to it.

Compare-against names the implementation whose reading is most useful at that
stage, not the one that is right. [`docs/working/coderef.md`](../working/coderef.md) says what each
checkout is and is not evidence of.

---

## The model branch

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| checkpoint load | core `UNETLoader` | [`comfyui_vendor_gaps.md`](comfyui_vendor_gaps.md) | `check_model_files.py` — a graph naming a file its loader cannot open goes red | LightX2V, for what the same int8 path does there |
| sigma shift | core `MiniMaxH3SigmaShift` | [`../../workflows/h3_config.py`](../../workflows/h3_config.py)'s `SAMPLING` note | `check_distill_settings.py` (shift and step count) | diffusers' named H3 scheduler |
| attention patch | core's `ModelAttentionBackend` on kitchen int8, the default since 2026-09-15; `attention.py`'s `MiniMaxH3SageAttention` on the sage arms | [`docs/attention/sol_attn.md`](../attention/sol_attn.md) | `check_attention_defaults.py` — by reachability, and values not presence | sglang (dense FA varlen); LightX2V for the kernel choice on this card |
| sparse attention | the vendored Sol node | [`docs/attention/sol_attn.md`](../attention/sol_attn.md) | same | `comfy-kitchen-kijai` (read-only; `comfy-kitchen-sol` until 2026-09-08), for sources that ship in no wheel |
| channel balance (off by default) | `channel_balance.py`, `MiniMaxH3ChannelBalance` | [`docs/attention/block49_quant_error.md`](../attention/block49_quant_error.md) | `check_channel_balance.py` — the fold is exact, RoPE-safe, off by default, and selects 45/48/49 on the shipped weights | the sage fork's `spike_h3_k_channel_balance.py`, the same fold graded on sage's kernel; `grade_channel_balance.py` on Sol's |
| chain order | the generator's wiring, `workflows/build_workflows.py::build_api` (`SageChainAssert` left every generated graph on 2026-09-17 and was deleted in 0.174.0) | [`custom_node_gaps.md`](custom_node_gaps.md) | `check_attention_defaults.py`, by reachability and the dense floor; Sol's own composition log at render time | — |
| step distillation | `MiniMaxH3PDDLoRA`, `pdd_math.py` | [`docs/distills/pdd.md`](../distills/pdd.md) | `check_pdd_sigmas.py`, `check_pdd_head_selection.py`, `check_distill_grid.py` | [`docs/research/pdd/2026-08-28_pdd_implementations.md`](../research/pdd/2026-08-28_pdd_implementations.md) — four other implementations |

## The conditioning branch

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| encoder load | `MiniMaxH3EncoderLoader` (`h3_encoder_loader.py`): core's own load plus the guards core lacks, on every generated graph. Graphs ship `h3_config.ENCODER_INT8`; the owner runs the bf16 pruned file, built from the release by `bench/convert_h3_bf16_encoder.py`. The AWQ adapter that was the alternate was deleted 2026-09-13 ([`docs/state/decisions.md`](../state/decisions.md)) | [`conditioning.md`](conditioning.md) section 0 | `check_h3_encoder_loader.py` | — |
| prompt structure | `prompt_bank/<id>.txt`, loaded by id through `workflows/prompts.py` (the generator's constants were literals until 2026-09-03) | `vendor_guides/` | `preflight_graph.py` grades sections, markers, labels — **reports, never refuses**; motion vocabulary is **nothing** | sglang's presentation stage |
| text encode | core `comfy/text_encoders/minimax.py` | [`docs/research/2026-08-21_official_weights_metadata.md`](../research/2026-08-21_official_weights_metadata.md); int8 against bf16 by token type: `../../data/results/2026-09-27_encoder_int8_vs_bf16_conditioning.json`; at the DiT: `bench/measure_encoder_quant_dit.py` | — | all four; they agree here |
| keyframes | `MiniMaxH3Conditioning`, `keyframe_canvas.py` (the geometry function only; its node was retired 0.173.0) | [`conditioning.md`](conditioning.md) | `check_conditioning_behaviour.py`, against core as reference | sglang |
| reference ingestion | the three `AppendRef*` nodes | [`references.md`](references.md) | `check_reference_runtime.py` — order, fps, duration, mono, policy | sglang, DiffSynth, diffusers |
| reference sizing | `reference_geometry.py`, `reference_report.py` (its pricing feeds `MiniMaxH3ReferenceConditioning`'s preview of both copies of every reference) | [`reference_sizing.md`](reference_sizing.md) | `check_reference_runtime.py`; the video model's copy is at vendor parity and the **encoder's separate copy, the default since 2026-10-03, is unmeasured** (`bench/refview2_arms.json`) | all three agree with the video model's copy |
| packing and positions | core `comfy/ldm/minimax/model.py` | [`conditioning.md`](conditioning.md) sections 3 and 4 | `check_reference_contracts.py` (all five contracts) | sglang; LightX2V matches core exactly |
| canvas and length | `MiniMaxH3Resolution`, `resolution.py`, `h3_rules.py` | [`geometry.md`](geometry.md) | `check_keyframe_canvas.py` for the keyframe canvas | diffusers' `resolve_canvas_size` |
| static pricing | `MiniMaxH3Preflight`, `bench/preflight_graph.py` | [`docs/method/checks.md`](../method/checks.md) | reports, never refuses; **nothing** asserts it stays wired | — |

## Sampling, decode, output

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| sigma schedule | core `BasicScheduler`, or the PDD node's own SIGMAS | [`docs/distills/pdd.md`](../distills/pdd.md) | `check_distill_grid.py` against the **vendor's published grid**, not a computed one | LightX2V's closed form |
| sampler | core `SamplerCustomAdvanced` | [`../../workflows/h3_config.py`](../../workflows/h3_config.py) | **nothing** reads the sampler name off a graph — the `euler`-for-PDD rule | sglang runs eta-0 Euler; we run a stochastic SDE |
| audio carry | core `comfy/model_sampling.py` | — | **nothing.** One schedule carries both streams; the other three run two | sglang, DiffSynth, diffusers |
| video decode | core `VAEDecode` | [`comfyui_vendor_gaps.md`](comfyui_vendor_gaps.md) | — | diffusers pins fp32; we ship fp16 |
| VAE precision | `MiniMaxH3VAEPrecision` | [`comfyui_vendor_gaps.md`](comfyui_vendor_gaps.md) | wired in one arm; **three implementations run more precisely than the shipped default** | diffusers, DiffSynth |
| audio decode | core `VAEDecodeAudio` | [`references.md`](references.md) | — | agreement is broad here |
| mux and write | `VHS_VideoCombine` | — | **nothing.** sglang re-probes the written file and fails on drift; we write and stop | sglang's output adapter |
| provenance | `MiniMaxH3ProvenanceStamp`, `substrate.py` | [`docs/working/capture_manifest_schema.md`](../working/capture_manifest_schema.md) | `check_provenance_stamp.py`, `check_capture_manifest.py` | — |

---

## Two facts that cut across every stage

**A rendered clip cannot A/B a numerical change.** The trajectory diverges
completely from any perturbation, on any sampler, at frame 0. The output of a
changed arm is a *different sample*, not a degraded one.
[`docs/method/eval_comparison.md`](../method/eval_comparison.md) owns the process; the root `AGENTS.md`
owns the rule. Compare knobs at the call, not at the output.

Two riders on that were established 2026-08-28 and now live with the process
they constrain, under *What a matched seed does not match* in
[`docs/method/eval_comparison.md`](../method/eval_comparison.md): the audio stream is not
matched by a shared seed when arms differ in canvas or length, and the first
run after a state change is not the arm's settled behaviour. That file owns
them; this row exists so nobody rediscovers them from the stage table.

## Masked video to video

The lane's own page is [`docs/masked_v2v/overview.md`](../masked_v2v/overview.md). These rows are the
stages a masked render adds to the song chain.

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| cuts, the pick, the match across shots | `subject_track.py` (`cut_scores`, `main_subject`, `follow`) | the module docstring; [`docs/masked_v2v/overview.md`](../masked_v2v/overview.md) for the limits | `check_subject_track.py`, with stand-ins for SAM 3: it cannot check that the features tell real people apart | core's `SAM3_VideoTrack` with a text prompt, for what it does across a cut; `coderef/sam3` for the caller-mask request |
| tracking inside a shot | core `SAM3_VideoTrack` on its `initial_mask` path, called from `subject_track.py` | core | nothing | `coderef/sam3` |
| the region and what is encoded | `video_mask.py` (`grow`, `token_mask`, `window`) | the module docstring; [`audio_freeze.md`](audio_freeze.md) section 4 | `check_video_mask.py` | core's `SetLatentNoiseMask` and mask resize, which the docstring says are not enough |
| the composite | `video_mask.py` (`changed_alpha`, `composite`) | [`audio_freeze.md`](audio_freeze.md) section 4 | `check_video_mask.py` | nothing |
| the mask through two samplers | `plate_restore.py`, in no shipped graph | the module docstring | `check_plate_restore.py`, on a stub model: not run on H3 | core's `KSamplerX0Inpaint` |
