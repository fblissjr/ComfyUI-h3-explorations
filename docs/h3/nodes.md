# Which nodes a render uses, and in what order

The node list for an H3 video graph: core's, this pack's, the ones to leave out, and the order on the model branch.
`workflows/build_workflows.py::build_api` is what the generator actually wires; read it over this page, and walk
`h3_config.graph_paths` for what a shipped graph carries. What each of our nodes does end to end, and where it
departs from the other implementations: [`custom_node_gaps.md`](custom_node_gaps.md). The canvas and length these
nodes take: [`geometry.md`](geometry.md).

## Core ComfyUI

| node | notes |
|---|---|
| Load Diffusion Model (`UNETLoader`) | the `fl2va` checkpoint for text to video and keyframes (it also takes reference stills), `ref2va` for references; the files are `h3_config.MODELS` |
| `VAELoader`, twice | the video VAE and the audio VAE are separate loaders |
| `ModelSamplingMiniMaxH3` (class `MiniMaxH3SigmaShift`) | the sigma shift; wired on every graph except a PDD graph at the default shift, where the PDD node builds its schedule from its own fused shift (`build_api`, the `pdd and sh == SIGMA_SHIFT` test). `bench/check_distill_settings.py` checks its value against the LoRA each graph loads |
| Model Attention Backend (`ModelAttentionBackend`) | the dense kernel under Sol on the default chain, "comfy kitchen attention" (`h3_config.DENSE_BACKEND_NODE`, `DEFAULT_DENSE_CHAIN`) |
| `RandomNoise`, `KSamplerSelect`, `BasicScheduler`, `BasicGuider`, `SamplerCustomAdvanced` | the custom-sampler stack; the defaults are `h3_config.SAMPLING`, with the reasoning beside it |
| `VAEDecode`, `VAEDecodeAudio`, then the video writer | video and audio decode separately |
| `MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo`, `CLIPLoader` | core's own conditioning and encoder load. Generated graphs wire this pack's replacements instead (below); core's nodes remain the reference a check compares ours against |

**Sampler choice is a cost choice too.** The default, `h3_config.SAMPLING`, is one model evaluation a step.
`heun`, `dpm_2` and the `2s`/`3s`/`res_Ns` families take two to six a step and multiply nearly all of a render's
time, decode being the small remainder.

## This pack's nodes on every generated graph

- **`MiniMaxH3EncoderLoader`** (`h3_encoder_loader.py`): core's own `CLIPLoader` load of
  `h3_config.MODELS["clip"]`, then a refusal if the load left parameters unpopulated or the tokenizer did not
  realize the release's marker ids, and a stamp of what core's preprocessing will do, read out of core, for
  preflight and the reference preview. It adapts no format and touches no weight. For a specific GPU, follow it
  with core's `SelectCLIPDevice`. Controlled by `bench/check_h3_encoder_loader.py`.
- **`MiniMaxH3Conditioning`** (`conditioning.py`) in place of core's `MiniMaxH3ImageToVideo` on the text-to-video
  and keyframe graphs; it owns the canvas ([`geometry.md`](geometry.md), "A canvas from a keyframe").
- **`MiniMaxH3ReferenceConditioning`** with the three `MiniMaxH3AppendRef*` nodes in place of core's
  `MiniMaxH3ReferenceToVideo` ([`references.md`](references.md)). After it runs, the conditioner shows on the
  node what each reference cost in both copies (the pricing in `reference_report.py`).
- **`MiniMaxH3Resolution`** (`resolution.py`) on text to video: a canvas by shape, with its cost.
- **`MiniMaxH3Preflight`** (`preflight.py`): prices the packed sequence before sampling and reports; it never
  refuses.
- **`MiniMaxH3Sol`** (`sol_attn_h3.py`, display name "MiniMax H3 Sparse Attention"): block-sparse attention, on in
  every shipped video graph (the exempt set is `bench/check_attention_defaults.py::SOL_EXEMPT_STEMS`). It must come
  after the dense node: it composes with the attention override it finds, and reversed you silently get the dense
  kernel only. Its settings are `h3_config.SOL_RECOMMENDED_CUDA`. [`../attention/sol_attn.md`](../attention/sol_attn.md)
  owns every Sol knob and number.

## This pack's nodes on some graphs

- **`MiniMaxH3SageAttention`** (`attention.py`, "MiniMax H3 SageAttention"): the dense kernel on the sage chain, in
  place of core's backend node, on graphs that declare why. It replaces every DiT attention forward with the sage
  fork's INT8 kernel and also registers an `optimized_attention_override`, which is what lets Sol compose with it.
- **`MiniMaxH3PDDLoRA`** and the other distill loaders: before the attention nodes
  ([`../distills/which_distill.md`](../distills/which_distill.md)).
- **`MiniMaxH3ProvenanceStamp`** (`provenance.py`), bench graphs only: a JSON sidecar under `output/provenance/` of what the settings resolved to,
  which `/history` cannot know (resolved sigmas, the Sol closure, node-pack heads, the snapped length and canvas),
  above all `n_sparse`, the sigma window intersected with the sampler's schedule. Wire `SIGMAS` from
  `BasicScheduler` and pass the sampler's `LATENT` through it, or it can run before sampling; it joins to
  `/history` on `graph_sha256`. The module docstring has its states and its two cautions: a well-provenanced number
  is not a verified one.
- **Per-block Sol error**: `sol_block_probe.py`, armed by the `H3_SOL_PROBE` environment variable rather than wired,
  read by `bench/read_sol_probe.py --record`. An armed render also runs the fallback on every Sol call, so its
  timings are void.

## Skip, with reasons

- **`MiniMaxH3MemoryEfficientSageAttentionPatch`** (KJNodes): the same job as the sage node on the same key; the last
  applied wins. Pick one.
- **`MiniMaxLowVRAMAttention`** (KJNodes): head chunking, which multiplies the attention launches by the chunk count
  for headroom that converts to wall time at a low ceiling on this card (`h3_config.SAGE_NODE`'s note, the head-chunk
  arms). Only for a render that otherwise will not fit.
- **`MiniMaxChunkFeedForward`** (KJNodes): at long lengths the attention peak is well above the FFN's, so it lowers a
  peak that does not bind; a short-clip feature. The arms are beside the head-chunk ones.
- **`PathchSageAttentionKJ`**: a process-wide sage switch with no per-model guard. Prefer the per-graph node.
- **`ModelPreviewOverrideKJ`**: a preview during sampling, in no generated graph (its decodes cost GPU time on the
  render it previews); add it by hand to watch one.
- **Core's `ResolutionSelector`**: it works from a megapixel target, which is not how H3 sizes a canvas.
- **Untested here**, no recommendation either way: `EasyCache`, `MiniMaxH3TurboLoRA`.

## Node order on the model branch

```
Load Diffusion Model (UNETLoader)
  -> LoRA / PDD loader               distilled arms only; before the patches
  -> ModelSamplingMiniMaxH3          anywhere before the fork; absent on a PDD graph at the default shift
  -> Model Attention Backend         core's dense kernel; the sage chain wires MiniMax H3 SageAttention here
  -> MiniMax H3 Sparse Attention     MiniMaxH3Sol, after the dense node
  -> BasicScheduler and BasicGuider  MODEL forks to both: rewire both
```

**MODEL forks to two consumers**, `BasicScheduler.model` and `BasicGuider.model`. Rewiring only the guider leaves
the scheduler reading sigmas off the unpatched model, and the render still succeeds, which is why that mistake
survives; the generated graphs drive both from one variable. A PDD graph that is not split has no `BasicScheduler`:
the PDD node emits the schedule, and only the guider reads MODEL. The order is checked statically, by reachability,
in `bench/check_attention_defaults.py`.
