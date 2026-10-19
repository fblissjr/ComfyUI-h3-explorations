# Where our custom nodes differ from every other H3 implementation

What the nodes in this pack do end to end, and where that differs from the vendor's serving path, two inference
engines, two model libraries and ComfyUI itself. The companion to [`comfyui_vendor_gaps.md`](comfyui_vendor_gaps.md),
which asks how native ComfyUI differs from the release. A catalogue, not an authority: each fact's owner is named
where it appears, and the owner wins. Which nodes a graph uses and in what order: [`nodes.md`](nodes.md).

**The answer.** The reference view the text encoder sees is the one choice of ours that most plausibly changes the
output, and it is built and unscored (§4.1). The attention patch is the sharpest structural divergence (§3). The rest
of the chain matches the other implementations, or differs only in cost.

Everything is a source read unless marked: *read* (the code at the cited path), *measured* (a record or a parsed
artifact header), *inference* (a conclusion, with its mechanism named so it can be refuted). Compared, as of the
2026-08-28 pass:

| implementation | what it is | revision |
|---|---|---|
| this pack | the nodes `H3ExplorationsExtension.get_node_list` registers (`nodes.py`) | working tree |
| native ComfyUI | `comfy_extras/nodes_minimax_h3.py`, `comfy/ldm/minimax/`, `comfy/text_encoders/minimax.py` | the installed checkout |
| `coderef/sglang` | the vendor's own serving path | `803b4fb31c` |
| `coderef/LightX2V` | inference engine; origin of the SLA work and the Turbo LoRAs | `5169278f` |
| `coderef/DiffSynth-Studio` | model library with a native H3 pipeline | working checkout |
| `coderef/diffusers` | model library with a native H3 pipeline | working checkout |

PDD is covered only by its place in the chain: [`../research/pdd/pdd_implementations.md`](../research/pdd/pdd_implementations.md)
compares implementations and [`../distills/pdd.md`](../distills/pdd.md) owns the node contract.

## 1. The node surface

The registered set is `bench/node_id_manifest.json`, kept by `bench/check_node_ids.py`; this page does not restate the
count. The class is the useful fact: **load-bearing** means a shipped render is wrong or absent without it,
**convenience** means a graph could be wired by hand instead, **instrumentation** means it catches or records
something.

| node | class | in shipped graphs |
|---|---|---|
| `MiniMaxH3Conditioning` | load-bearing | yes |
| `MiniMaxH3ReferenceConditioning` | load-bearing | yes |
| `MiniMaxH3AppendRefImage` / `AppendRefVideo` / `AppendRefAudio` | load-bearing | yes |
| `MiniMaxH3EncoderLoader` | load-bearing (it refuses a bad load) | yes |
| `MiniMaxH3Sol` | load-bearing | yes |
| `MiniMaxH3SageAttention` | load-bearing on the sage chain | graphs that declare why |
| `MiniMaxH3PDDLoRA` | load-bearing on PDD arms | yes |
| `MiniMaxH3Resolution` | convenience | yes |
| `MiniMaxH3Preflight` | instrumentation | yes |
| `MiniMaxH3VAEPrecision` | instrumentation | the fp32 probe arm |
| `MiniMaxH3ProvenanceStamp` | instrumentation | bench only |
| `MiniMaxH3ChannelBalance` | instrumentation: an experiment lever, off by default | no |

`MiniMaxH3ChannelBalance` folds a per-channel q/k rebalancing into the norm weights of the blocks whose
`k_norm.weight` is lopsided (45, 48, 49 on the shipped checkpoint), so both INT8 kernels quantize them with less error
at no render-time cost; the mechanism is [`../attention/block49_quant_error.md`](../attention/block49_quant_error.md).
`keyframe_canvas.py::resolve_keyframe_geometry` and `marker_arms.py` stay as libraries after their nodes were retired.
Nodes retired in 0.173.0 and 0.174.0 (`MiniMaxH3KeyframeCanvas`, `MiniMaxH3ReferenceFit`, `MiniMaxH3ReferenceVideoFit`,
`MiniMaxH3MarkerArm`, `SageChainAssert`, `MiniMaxH3ReferenceReport`, `MiniMaxH3QuantObserve`, `MiniMaxH3VSAAttention`)
and earlier ones (`MiniMaxH3AWQEncoderLoader`, `MiniMaxH3SolAttnCurve`) are in `CHANGELOG.md` and git; prose elsewhere
that names them describes the repo before.

**Two asymmetries in what is asserted.** `MiniMaxH3Preflight`, in every shipped graph, has no dedicated check and
nothing asserts it stays wired; the most heavily asserted node, `MiniMaxH3ProvenanceStamp`, is on bench graphs only.
Coverage grew where the work happened, not where the renders are.

## 2. End to end, per mode

All three canonical graphs share one model branch and differ in the conditioner, the media loaders and the
checkpoint. Every arrow is *read*.

### The model branch

```
UNETLoader ─MODEL─> ModelSamplingMiniMaxH3 ─> dense attention ─> MiniMaxH3Sol ─┬─> BasicScheduler
                    (core; clone, object_patch   core's Model Attention  (ours; chains     └─> BasicGuider
                     model_sampling, shift_v/a)  Backend, or our sage     its override onto
                                                 node on the sage chain   the dense one)
```

**The order is load-bearing.** Sol composes with the attention override it finds; reversed, it overwrites ours and
you silently get the dense kernel only. With the sage node, its patched forward calls its kernel directly and never
consults `optimized_attention`, so only Sol's gate can hand those calls back. The order is checked statically, by
reachability, in `bench/check_attention_defaults.py`.

### t2v

```
MiniMaxH3EncoderLoader ─CLIP─┐   MiniMaxH3Resolution ─w,h,len─┐
                             ▼                                ▼
          MiniMaxH3Conditioning (canvas=explicit)  <── VAELoader (video)
                 │  CONDITIONING [1,L,5120] + minimax_token_tags [L]
                 │  LATENT  Nested(video, audio), fp32
                 ▼
          MiniMaxH3Preflight ── prices the packed sequence, reports only
                 ├─CONDITIONING─> BasicGuider ─> SamplerCustomAdvanced
                 └─LATENT────────────────────────────────┘
                     packed: [ text | audio | video ]
             VAEDecode (unbind[0])     VAEDecodeAudio (unbind[-1])  -> video writer
```

### fl2va

The same spine. Two `LoadImage` feed `MiniMaxH3Conditioning` as `first_frame` / `last_frame`; it resizes them,
VAE-encodes each, and attaches them as `minimax_keyframes`, pinned at the condition timestep and re-injected every
step. The packed layout gains two condition spans ahead of the audio rows.

### ref2va

```
LoadImage ─> MiniMaxH3AppendRefImage ─> MiniMaxH3AppendRefImage ─REFERENCES─┐
             (a frozen dataclass is appended; no pixels move here)          ▼
                                          MiniMaxH3ReferenceConditioning
                                          <-VAELoader(video) <-VAELoader(audio)
             packed: [ text | ref | ref | audio | video ]
```

The two sizing paths are separate, and this is the most consequential thing our nodes do differently (§4.1): stage
one (`reference_geometry.py::fit_reference_image`) sets what the VAE and the DiT see; with `qwen_view=separate`, the
default, a second view built from the source goes to the text encoder alone
([`reference_sizing.md`](reference_sizing.md)).

### Reading traps in the shipped JSON

1. **fl2va's `width`/`height` are inert** under `canvas: "from_keyframe"` (`conditioning.py::execute`), yet the JSON
   carries an authoritative-looking value that overstates the video rows.
2. **Only the API form ships**, and a disabled node is simply absent.

## 3. Against native ComfyUI

| native | ours | relationship | the difference that matters |
|---|---|---|---|
| `EmptyMiniMaxH3LatentAV` | absorbed into both conditioners | replaces | canvas and latent cannot disagree |
| `MiniMaxH3ImageToVideo` | `MiniMaxH3Conditioning` | replaces | refuses an empty prompt; a lone last frame anchors instead of being cover-cropped; canvas derived from the anchor; length snapped then ceiling-checked |
| `MiniMaxH3ReferenceToVideo` | `MiniMaxH3ReferenceConditioning` + three appends | replaces | sockets become an ordered tuple; suffix pairing becomes ownership; frame rate normalized from loader metadata; more than two channels refused; soundtrack capped; latent grid read off the VAE rather than the pixels |
| `MiniMaxH3AddGuide` | none | — | arbitrary-frame guides are reachable only through core, and chain onto ours unchanged |
| `MiniMaxH3SigmaShift` | none | consumed | a core node, used as-is |
| `ModelAttentionBackend` | none | consumed | the default dense kernel |
| `optimized_attention` / `_override` | `MiniMaxH3SageAttention` | replaces, on the sage chain | per-module forward object patch |
| `CLIPLoader` | `MiniMaxH3EncoderLoader` | wraps | core's own load, then refuses a silently incomplete load or wrong marker ids, and stamps what core's preprocessing will do (§5.1) |
| one `vae_dtype` | `MiniMaxH3VAEPrecision` | wraps | encode and decode split apart |
| `LoraLoaderModelOnly` | `MiniMaxH3PDDLoRA` | supplements | core's loader applies the backbone and silently skips the mechanisms that are not weight patches |
| width/height ints | `MiniMaxH3Resolution` | supplements | the trained family by shape, with its cost |
| nothing | `MiniMaxH3Preflight`, `MiniMaxH3ProvenanceStamp` | no counterpart | — |

**The sage node's attention bypass is the sharpest divergence** (*read*). The native H3 path honors
`optimized_attention_override` fully: it passes `transformer_options` down and boxes q/k/v as containers. The sage
node's object patch bypasses that layer, and with it the global backend flags, the per-call override, the container
protocol and sage's own fallback; `attention.py::make_sage_override` exists to re-enter it. `optimized_attention_for_device`
is not lost, because the H3 DiT never called it. The default chain's dense kernel is core's own node and has none of
this.

**The five reference contracts** belong to core's `MiniMaxH3ReferenceToVideo` and are owned by
[`../research/conditioning_nodes.md`](../research/conditioning_nodes.md). Core holds all five, the fifth on both
branches of `CLIP.encode_from_tokens_scheduled` since `6bfaacc6` (2026-09-20), and ours inherits that seam exactly.

## 4. Against the vendor and the engines

### 4.1 Changes the output

1. **The reference view the text encoder sees.** sglang, DiffSynth and diffusers all feed one prepared still, at a
   2048 short edge with upscaling, to both the VAE and Qwen. Our video-model copy matches; the text encoder's is
   `qwen_view=separate` at `h3_rules.REF_QWEN_SHORT_EDGE` by default since 2026-10-03, the owner's choice on a few
   clips at one seed (`docs/state/decisions.md`). It was first the default on 2026-08-27, as the fix for a
   two-speaker scene whose dialogue was misattributed after upscaled references crowded the prompt out of its
   segment, *priced, not proven* on one render (CHANGELOG 0.82.0). The ablation that would judge it is built and
   unscored: `bench/refview2_arms.json`, six arms on five scene graphs (`h3_config.REFVIEW2_SCENES`). **The strongest
   open candidate on this page.**
2. **Video VAE precision.** DiffSynth loads the release fp32 file; diffusers since `51a454be9` (#14754) keeps the
   encoder, `quant_conv`, the norms and the LayerScale scales in fp32 and decodes in the pipeline dtype
   (`coderef/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py::AutoencoderKLMiniMaxH3._keep_in_fp32_modules`;
   [`../working/coderef.md`](../working/coderef.md), "diffusers"). The canonical graphs load
   `h3_config.MODELS["video_vae"]` and wire no `MiniMaxH3VAEPrecision`; the `h3_probe_ref_vae_encoder_fp32*` arms
   against their `..._fp16*` controls do, and `data/results/2026-08-21_vae_encoder_precision.json` already grades the
   encoder half at the call. Whether the shipped default should follow is a decision
   ([`comfyui_vendor_gaps.md`](comfyui_vendor_gaps.md), gap 8).
3. **Condition latents are the posterior mean, not a seeded sample**, and **the condition-noise draw differs while
   the recipe matches** ([`reference_sizing.md`](reference_sizing.md), "The vendor image path").
4. **The audio stream rides the video schedule** (*inference*). Core carries one schedule and scales the audio latent
   by a constant, converting the velocity back inside the DiT; sglang, DiffSynth and diffusers run two independent
   schedules. The conversion is exact for a first-order update. Every graph but the two FastH3 ones steps with Euler
   (`h3_config.SAMPLING`), the exact case; `h3_config.FASTH3_SAMPLER` (`res_multistep`) is the one higher-order
   sampler left, whose corrections are built from video-schedule quantities and applied to audio rows too. Nothing
   covers it, and it is cheap to check at the call.
5. **The SLA router does not cover what the SLA LoRA was distilled on** (*measured*, from the artifact header). The
   Turbo-SLA LoRA carries modules for the fifty DiT blocks **and the token refiner**; the retired
   `MiniMaxH3SLARouter` patched the blocks only (`docs/state/open_experiments.md` #20). Also *read*: the distillation
   pooled queries in larger blocks than our router did, and the vendor kernel quantizes q/k/v where ours was bf16;
   that operator is not buildable on this box.

The sampler itself is no longer a divergence: the base runs Euler, which matches sglang's eta-0 Euler in kind (owner,
2026-10-05; `workflows/h3_config.py::SAMPLING`).

### 4.2 Changes only the cost

Sequence parallelism, AdaLN caching, CUDA graphs and step caching are sglang's and absent here;
[`../research/sglang_comparison.md`](../research/sglang_comparison.md) owns and prices that list. LightX2V adds block
prefetch, tensor and sequence parallelism, and block-granular text-encoder offload, the last being the shape of a fix
for the host-memory incident during a bridge load; we have no equivalent. Two apparent gaps are savings (*read*): we
omit sglang's padding tail, whose rows sit in their own attention segment, and we resample reference audio once where
sglang does it twice.

### 4.3 Where the agreement is worth banking

All four implementations agree character for character on the reference label rules, including a sounded video's
audio label immediately before its video label: no system prompt, no chat template, the same encoder layer, no
pooling, no truncation, the same tag convention. The VAE normalization statistics are bit-identical between ComfyUI
and DiffSynth (*measured*: the decimal text differs, the doubles do not). H3's decode tiling is not a silent ComfyUI
branch: the H3 VAE hard-wires it, matching the release. Neither library is independent evidence about the seven
markers: both inherit the release tokenizer, so they are not two more votes.

## 5. More than one way to do it

### 5.1 Text-encoder loader

Every shipped graph wires `MiniMaxH3EncoderLoader` with `h3_config.MODELS["clip"]`: core's own `CLIPLoader` load,
a refusal if the load left parameters unpopulated or the marker ids are not the release's, and a stamp of what core's
preprocessing will do (`h3_encoder_loader.native_encoder_contract`, read out of core). The generator refuses any file
outside `workflows/h3_config.py::CORE_LOADED_ENCODERS` (*read*). It is a stock load path with the guards core lacks,
on the encoder closest to the bf16 release at the layer that matters. The W4A16 AWQ adapter and the `encoder` policy
that went with it were deleted on 2026-09-13 (`docs/state/decisions.md`).

### 5.2 Attention

Two dense chains under Sol (core's kitchen backend, the default, and the sage node in its rotated mode;
`h3_config.DEFAULT_DENSE_CHAIN` and the note beside it), plus stock attention on the baseline graph.
`bench/check_attention_defaults.py` grades by reachability from the output node rather than node presence, pins values
rather than presence, derives its exempt classes and asserts each exemption is necessary.

### 5.3 Step reduction

Three loaders that are silently non-interchangeable (*read*). The turbo-pack file cannot be applied by the stock
loader at all. The stock loader on a PDD file applies the backbone and skips the two mechanisms that are not weight
patches, producing an undistilled render that looks fine. PDD graphs own their schedule and carry no
`BasicScheduler`. See [`../distills/pdd.md`](../distills/pdd.md).

### 5.4 Conditioning, resolution, VAE, caching

Our two conditioners replace core's everywhere; `image_policy` is `comfy` on every reference graph and
`video_policy` `comfy` on all but the `release` probe arm (walk `h3_config.graph_paths`). `MiniMaxH3Resolution` is
convenience over typing two integers. The VAE-precision and cache axes are single-graph probes with committed
measurements. A conditioning-encode cache, if one is ever built, should hold the joint encode only: a peer pack that
cached separately encoded sections reverted it after reported generation distortion
(`coderef/ComfyUI-UtilsCollection` `d1921ae`; [`../working/coderef.md`](../working/coderef.md) describes the pack).
Output is not an axis: every graph wires the same writer with every input constant but the filename prefix.

## 6. What is enforced by nothing

Candidates for the standing audit in [`../method/checks.md`](../method/checks.md), not additions to it.

| requirement | status |
|---|---|
| Sol's live-module surgery is undone when the node is bypassed | **nothing.** It rebinds module attributes and registers hooks outside the patcher, keyed on object identity |
| a PDD arm is consumed by `euler` | **nothing**; already in `checks.md` |
| `MiniMaxH3Preflight` stays wired in shipped graphs | **nothing** |
| the attention nodes' order is what the graph contains | `check_attention_defaults.py`, statically |
| the sigma tail lands outside Sol's window | **nothing.** The node logs the window and never sees the sigmas |
| the audio stream's higher-order sampler terms are valid on the video schedule | **nothing** (§4.1) |
| `qkv_proj` row order matches the checkpoint's convention | **nothing on any side.** The same key name denotes two row orders across implementations; our converter targets the right one on the strength of a comment |
| `_SPANS` in `sol_attn_h3.py` stays bounded over a long-lived server | **nothing.** Keyed on `id(position_ids)`, one entry per distinct packed layout, each keeping its layout alive so the id cannot be recycled, and nothing evicts; growth unmeasured (flagged 2026-09-10) |

**Noise for the video and audio streams comes from one seeded generator consumed in order**, so the audio noise
depends on the video latent's element count: a matched-seed A/B whose arms differ in canvas or length is not matched
on the audio stream. [`../method/eval_comparison.md`](../method/eval_comparison.md) owns the A/B process, under "What a
matched seed does not match".

## 7. Actionable, cheapest first

1. **Fix the stale generator comment** in `workflows/build_workflows.py` that says one turbo arm ships "sage on and
   Sol absent", reasoning the same file retracts later; the emitted graph carries live Sol (*measured* 2026-08-28). No
   check reads generator comments.
2. **Decide the shipped VAE encode precision** against the graded result (§4.1, item 2).
3. **Record the SLA router asymmetry** against `docs/state/open_experiments.md` #20, now that the LoRA's key set
   confirms it.
4. **Score the reference-view ablation**, which is what turns §4.1's first item from priced into proven.

## 8. What this pass did not examine

Static reads only: nothing was rendered and no GPU was used, except where marked *measured*. Not examined: whether
any divergence here is visible in output; the single-frame path, which is parked; the audio-refine and looping packs;
and the bench harnesses except where a check was cited.
