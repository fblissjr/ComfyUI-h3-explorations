# One render, stage by stage: code, owner, guard, reference

last updated: 2026-08-28

The cross-index. For each stage of a render: **our code**, the **document that
owns** it, the **check that would go red** if it broke, and the
**implementation to compare against** when you need to know what it should do.

**Written by a person. Not generated** — the generator that builds
[`index.md`](index.md) never touches this file.

**"nothing" in the guard column is the useful entry.** It is not a to-do list;
CLAUDE.md's rule is that a new check needs a drift instance the existing gates
provably could not have caught. It is there so nobody mistakes an unguarded
stage for a guarded one. The standing audit is
[`../checks.md`](../checks.md)'s uncontrolled-requirements table, which is the
authority; this column routes to it.

Compare-against names the implementation whose reading is most useful at that
stage, not the one that is right. [`references.md`](references.md) says what each
checkout is and is not evidence of.

---

## The model branch

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| checkpoint load | core `UNETLoader` | [`../comfyui_vendor_gaps.md`](../comfyui_vendor_gaps.md) | `check_model_files.py` — a graph naming a file its loader cannot open goes red | LightX2V, for what the same int8 path does there |
| sigma shift | core `MiniMaxH3SigmaShift` | [`../../workflows/h3_config.py`](../../workflows/h3_config.py)'s `SAMPLING` note | `check_distill_settings.py` (shift and step count) | diffusers' named H3 scheduler |
| attention patch | `attention.py`, `MiniMaxH3SageAttention` | [`../SOLATTN.md`](../SOLATTN.md) | `check_attention_defaults.py` — by reachability, and values not presence | sglang (dense FA varlen); LightX2V for the kernel choice on this card |
| sparse attention | the vendored Sol node | [`../SOLATTN.md`](../SOLATTN.md) | same | `comfy-kitchen-sol`, for sources that ship in no wheel |
| chain order | `SageChainAssert` | [`../custom_node_gaps.md`](../custom_node_gaps.md) | itself, at call time — and **nothing** asserts it stays wired | — |
| step distillation | `MiniMaxH3PDDLoRA`, `pdd_math.py` | [`../h3_pdd.md`](../h3_pdd.md) | `check_pdd_sigmas.py`, `check_pdd_head_selection.py`, `check_distill_grid.py` | [`../research/pdd/pdd_implementations.md`](../research/pdd/pdd_implementations.md) — four other implementations |

## The conditioning branch

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| encoder load | core `CLIPLoader`; `MiniMaxH3AWQEncoderLoader` is the alternate | [`../h3_awq_encoder.md`](../h3_awq_encoder.md) | `check_h3_awq_encoder.py` | — |
| prompt structure | the generator's prompt constants | `internal/official_prompt_guides/` | `preflight_graph.py` grades sections, markers, labels — **reports, never refuses**; motion vocabulary is **nothing** | sglang's presentation stage |
| text encode | core `comfy/text_encoders/minimax.py` | [`../research/official_weights_metadata.md`](../research/official_weights_metadata.md) | — | all four; they agree here |
| keyframes | `MiniMaxH3Conditioning`, `keyframe_canvas.py` | [`../h3_conditioning_end_to_end.md`](../h3_conditioning_end_to_end.md) | `check_conditioning_behaviour.py`, against core as reference | sglang |
| reference ingestion | the three `AppendRef*` nodes | [`../h3_references.md`](../h3_references.md) | `check_reference_runtime.py` — order, fps, duration, mono, policy | sglang, DiffSynth, diffusers |
| reference sizing | `reference_fit.py`, `reference_geometry.py` | [`../h3_references.md`](../h3_references.md) | `check_reference_fit.py`; the **encoder view split is three-against-one and unmeasured** | all three agree against us |
| packing and positions | core `comfy/ldm/minimax/model.py` | [`../h3_geometry_and_nodes.md`](../h3_geometry_and_nodes.md) | `check_reference_contracts.py` (all five contracts) | sglang; LightX2V matches core exactly |
| canvas and length | `MiniMaxH3Resolution`, `resolution.py` | [`../h3_resolutions.md`](../h3_resolutions.md) | — | — |
| static pricing | `MiniMaxH3Preflight`, `bench/preflight_graph.py` | [`../checks.md`](../checks.md) | reports, never refuses; **nothing** asserts it stays wired | — |

## Sampling, decode, output

| stage | our code | owner | guard | compare against |
|---|---|---|---|---|
| sigma schedule | core `BasicScheduler`, or the PDD node's own SIGMAS | [`../h3_pdd.md`](../h3_pdd.md) | `check_distill_grid.py` against the **vendor's published grid**, not a computed one | LightX2V's closed form |
| sampler | core `SamplerCustomAdvanced` | [`../../workflows/h3_config.py`](../../workflows/h3_config.py) | **nothing** reads the sampler name off a graph — the `euler`-for-PDD rule | sglang runs eta-0 Euler; we run a stochastic SDE |
| audio carry | core `comfy/model_sampling.py` | — | **nothing.** One schedule carries both streams; the other three run two | sglang, DiffSynth, diffusers |
| video decode | core `VAEDecode` | [`../comfyui_vendor_gaps.md`](../comfyui_vendor_gaps.md) | — | diffusers pins fp32; we ship fp16 |
| VAE precision | `MiniMaxH3VAEPrecision` | [`../comfyui_vendor_gaps.md`](../comfyui_vendor_gaps.md) | wired in one arm; **three implementations run more precisely than the shipped default** | diffusers, DiffSynth |
| audio decode | core `VAEDecodeAudio` | [`../h3_references.md`](../h3_references.md) | — | agreement is broad here |
| mux and write | `VHS_VideoCombine` | — | **nothing.** sglang re-probes the written file and fails on drift; we write and stop | sglang's output adapter |
| provenance | `MiniMaxH3ProvenanceStamp`, `substrate.py` | [`../capture_manifest_schema.md`](../capture_manifest_schema.md) | `check_provenance_stamp.py`, `check_capture_manifest.py` | — |

---

## Two facts that cut across every stage

**A rendered clip cannot A/B a numerical change.** The trajectory diverges
completely from any perturbation, on any sampler, at frame 0. The output of a
changed arm is a *different sample*, not a degraded one.
[`../eval_comparison.md`](../eval_comparison.md) owns the process; CLAUDE.md
owns the rule. Compare knobs at the call, not at the output.

Two riders on that were established 2026-08-28 and now live with the process
they constrain, under *What a matched seed does not match* in
[`../eval_comparison.md`](../eval_comparison.md): the audio stream is not
matched by a shared seed when arms differ in canvas or length, and the first
run after a state change is not the arm's settled behaviour. That file owns
them; this row exists so nobody rediscovers them from the stage table.
