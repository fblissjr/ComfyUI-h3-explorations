# Where our custom nodes differ from every other H3 implementation

last updated: 2026-09-13 (the AWQ loader rows, the `encoder` policy and §4.1's reference-view row corrected after the owner's decisions of that day, `docs/wiki/decisions.md`; 2026-09-10 added one §5.4 sentence and one §6 row; everything else is the 2026-08-28 pass)

The companion to [`comfyui_vendor_gaps.md`](comfyui_vendor_gaps.md). That file
asks how native ComfyUI differs from the MiniMax release. This one asks the
adjacent question nobody had written down: **what do the nodes in this pack
actually do, end to end, and where does that differ from the vendor's serving
path, from two inference engines, from two model libraries, and from ComfyUI
itself.**

**This file is a snapshot, not an authority.** Every fact in it is owned by
another document, named where it appears. If this file disagrees with an owner,
the owner is right and this one is stale. Regenerate it by re-reading the
owners, not by editing it in place.

**Everything here is a source read unless it says otherwise.** Per CLAUDE.md, a
source read is an inference, not a run. Claims are labelled inline:
*read* (the code at the cited path), *measured* (a record in `bench/results/`
or an artifact header parsed for this document), *inference* (a conclusion
drawn, with the mechanism named so it can be refuted). No number appears below
unless a different plausible value would change the reader's next action.

What was compared, and at what revision:

| implementation | what it is | revision |
|---|---|---|
| this pack | the nodes `H3ExplorationsExtension.get_node_list` registers, in this pack's `nodes.py` | working tree, 2026-08-28 |
| native ComfyUI | `comfy_extras/nodes_minimax_h3.py`, `comfy/ldm/minimax/`, `comfy/text_encoders/minimax.py` | the installed checkout |
| `coderef/sglang` | the vendor's own serving path | `803b4fb31c` |
| `coderef/LightX2V` | inference engine; origin of the SLA work and the Turbo LoRAs | `5169278f` |
| `coderef/DiffSynth-Studio` | model library with a native H3 pipeline | working checkout |
| `coderef/diffusers` | model library with a native H3 pipeline | working checkout |

PDD is deliberately not covered here beyond its node's place in the chain. It
has its own comparison at
[`research/pdd/pdd_implementations.md`](research/pdd/pdd_implementations.md),
and [`h3_pdd.md`](h3_pdd.md) owns the converter and node contract.

---

## 1. The node surface

The registered nodes are what this pack's extension class returns, recorded in
`bench/node_id_manifest.json` by `bench/check_node_ids.py`; this page does not
restate the count. They fall into three classes, and the class is the useful
fact: **load-bearing** means a shipped render is wrong or absent
without it, **convenience** means the graph could be wired by hand instead, and
**instrumentation** means it exists to catch or record something.

| node | class | wired in shipped graphs |
|---|---|---|
| `MiniMaxH3Conditioning` | load-bearing | yes |
| `MiniMaxH3ReferenceConditioning` | load-bearing | yes |
| `MiniMaxH3AppendRefImage` / `AppendRefVideo` / `AppendRefAudio` | load-bearing | yes |
| `MiniMaxH3SageAttention` | load-bearing | yes |
| `SageChainAssert` | load-bearing (it raises) | yes |
| `MiniMaxH3PDDLoRA` | load-bearing on PDD arms | yes |
| `MiniMaxH3Resolution` | convenience | yes |
| `MiniMaxH3KeyframeCanvas` | convenience | **no** |
| `MiniMaxH3ReferenceFit` | convenience, **DEPRECATED 2026-08-28** | **no** |
| `MiniMaxH3ReferenceVideoFit` | convenience | **no** |
| `MiniMaxH3Preflight` | instrumentation | yes |
| `MiniMaxH3VAEPrecision` | instrumentation | the fp32 probe arm, not the canonical graphs |
| `MiniMaxH3ProvenanceStamp` | instrumentation | bench only |
| `MiniMaxH3MarkerArm` | instrumentation | **no** |
| `MiniMaxH3EncoderLoader` | load-bearing (it refuses a bad load) | yes |
| `MiniMaxH3ReferenceReport` | instrumentation | **no** (a UI node; the conditioner's preview carries the same text) |
| `MiniMaxH3ChannelBalance` | instrumentation (an experiment lever: off by default, changes numerics only when switched on) | **no** |

The registered nodes wired by no shipped graph are not dead code, and the
distinction matters:

- `MiniMaxH3KeyframeCanvas` was **folded into its consumer** —
  `keyframe_canvas.py::resolve_keyframe_geometry` is called by
  `MiniMaxH3Conditioning` on every keyframe render. Only the node wrapper is
  unwired.
- `MiniMaxH3ReferenceFit` and `MiniMaxH3ReferenceVideoFit` were superseded by
  the append-ref chain and kept registered so externally saved graphs still
  load. **`MiniMaxH3ReferenceFit` carries `is_deprecated=True` since
  2026-08-28**, so it no longer appears in the node picker while saved graphs
  that wire it keep loading. Three reasons it is not merely redundant: chaining
  it in front of the append resamples TWICE, it hardcodes `size_policy="max"`
  and cannot express `match`, and it has no `qwen_view`, so it cannot give
  the text encoder a view separate from the VAE's. `MiniMaxH3ReferenceVideoFit`
  is NOT deprecated -- it is unwired but still live as the reporter
  `bench/check_ref_video_prediction.py` grades core's behaviour against.
- `MiniMaxH3AWQEncoderLoader` **was deleted on 2026-09-13** with the AWQ
  lane's code (`docs/wiki/decisions.md`). Until then this row called it a
  format adapter with no consumer but live code, read by preflight and the
  config. See §5.1.
- `MiniMaxH3MarkerArm` is a research instrument.
- `MiniMaxH3ChannelBalance` (added 2026-09-14) folds a per-channel q/k
  rebalancing into the norm weights of the blocks whose `k_norm.weight` is
  lopsided (45, 48, 49 on the shipped checkpoint), so both INT8 attention
  kernels quantize them with less error at no render-time cost. Its combo is
  **off by default** and no graph wires it: an experiment under
  `docs/SOLATTN.md`'s decision standard, with the mechanism in
  `docs/h3_block49_quant_error.md`. Registered so a probe
  graph can switch it on without a code change.
- **`MiniMaxH3SolAttnCurve` was deleted on 2026-08-31** and is no longer in
  this table. It supplied a `hilbert` token ordering by rebinding
  `morton_perm` on the vendored Sol node; that node stopped being loaded on
  2026-08-30, so the rebind patched nothing and its `execute` could only
  raise. `MiniMaxH3SolAttn` owns the Morton code and offers `hilbert` in its
  own `morton_curve` combo.

**The Sol node every shipped graph wires is this repo's own file.**
`ComfyUI-SolAttn-cuda/sol_attn_minimax.py` is a symlink to
`vendor/sol_attn_minimax.py`, and that pack's `__init__.py` only re-exports the
entrypoint (*read*, verified directly). So `vendor/` is live production code on
every render, not a reference copy. That pack's own docstring still explains
"why this is not vendored into ComfyUI-h3-explorations" — prose that the symlink
contradicts. Cosmetic, but it is the kind of sentence that costs a session.

### Two asymmetries in what is asserted

The two nodes present in every shipped graph — `SageChainAssert` and
`MiniMaxH3Preflight` — have no dedicated check, and nothing asserts they stay
wired. The two most heavily asserted, `MiniMaxH3ProvenanceStamp` and
`MiniMaxH3MarkerArm`, are wired in bench graphs and none respectively. Coverage
has grown where the work happened, not where the renders are.

*2026-09-17:* `SageChainAssert` is no longer in any generated graph (owner); the
paragraph above describes the graphs before that.

---

## 2. End to end, per mode

All three canonical graphs share one spine and differ in three places: the
conditioner, the presence of the media loaders, and the checkpoint. Diagrams
condensed from the full trace; every arrow is *read*.

### The model branch, common to all three

```
UNETLoader ──MODEL──> MiniMaxH3SigmaShift ──> MiniMaxH3SageAttention
                      (core node)              (ours)
                          │                        │
              clone + object_patch          clone + one object_patch per DiT
              model_sampling (ModelSamplingAV)   block on .attn.forward,
              + transformer_options[shift_v/a]   + transformer_options
                                                   [optimized_attention_override]
                                                        │
                                                        ▼
                                              SolAttnMiniMax (vendored)
                                       clone; re-registers each sage forward
                                       wrapped in a compose gate; chains its
                                       override with sage as `previous`
                                                        │
                                                        ▼
                                              SageChainAssert (ours)
                                     two-sided call-time probe; raises
                                                        │
                                     ┌──────────────────┴─────────────┐
                                     ▼                                ▼
                              BasicScheduler                    BasicGuider
```

**The order is load-bearing in both directions** (*read*). Sage's patched
forward calls its kernel directly and never consults `optimized_attention`, so
only Sol's gate can hand those calls back; and the `previous` chain inverts if
sage is applied second, making Sol an unreachable fallback. `SageChainAssert`
is what proves it at call time, with a probe sized so one call must reach sage
and one must not — `None` counts as positive evidence because the probes run on
fresh threads.

### t2v

```
CLIPLoader ─CLIP─┐        MiniMaxH3Resolution ─w,h,len─┐
                 ▼                                     ▼
          MiniMaxH3Conditioning (canvas=explicit)  ←── VAELoader (video)
                 │  CONDITIONING [1,L,5120] + minimax_token_tags [L]
                 │  LATENT  Nested(video, audio), fp32
                 ▼
          MiniMaxH3Preflight ── prices the packed sequence, reports only
                 │
                 ├─CONDITIONING─> BasicGuider ─> SamplerCustomAdvanced
                 └─LATENT────────────────────────────────┘
                                    │
              packed: [ text | audio | video ]
                                    │
             ┌──────────────────────┴────────────────────┐
             ▼                                           ▼
      VAEDecode (unbind[0])                  VAEDecodeAudio (unbind[-1])
             └──────────────> VHS_VideoCombine <─────────┘  → mp4
```

### fl2va

Same spine. Two `LoadImage` feed `MiniMaxH3Conditioning` directly as
`first_frame` / `last_frame`; the conditioner resizes them, VAE-encodes each,
and attaches them as `minimax_keyframes` pinned at the condition timestep and
re-injected every step. Packed layout gains two condition spans ahead of the
audio rows.

### ref2va

```
LoadImage ─> MiniMaxH3AppendRefImage ─> MiniMaxH3AppendRefImage ─REFERENCES─┐
             (a frozen dataclass is appended; no pixels move here)          ▼
                                          MiniMaxH3ReferenceConditioning
                                          ←VAELoader(video) ←VAELoader(audio)
             packed: [ text | ref | ref | audio | video ]
```

**The two sizing paths are genuinely separate, and this is the single most
consequential thing our nodes do differently from everyone else** (§4.1).
Stage one (`reference_fit.py::fit_reference_image`) sets the geometry the VAE
and the DiT see. Stage two (`qwen_view_size`) builds a *second* view from the
source for the text encoder alone, when `qwen_view=separate`; since 2026-09-13
the default is `shared`, one copy for both, and `MiniMaxH3ReferenceReport`
draws what each reader gets.

### Reading traps in the shipped JSON

Three values in the shipped graphs do not mean what they appear to mean. None
is a bug; all three will mislead anyone pricing a render from the file.

1. **fl2va's `width`/`height` are inert.** The graph carries a wide canvas
   beside `canvas: "from_keyframe"`, which routes to the anchor-derived canvas
   instead (`conditioning.py::execute`). The tooltip says so; the JSON still
   carries an authoritative-looking value, and it overstates the video rows.
2. **`video_policy: "encoder"` no longer exists.** Until 2026-09-13 most graphs
   carried it and it was inert: core's `CLIPLoader` stamped no
   `_h3_encoder_contract`, so `reference_geometry.py`'s `effective_policy`
   resolved it to `comfy`, and only a log line reported the substitution. The
   rebuilt graphs carry `comfy`, which is what ran; the value and the function
   are gone.
3. **UI and API forms were not node-for-node identical** (resolved 2026-09-14:
   only the API form ships, and a disabled node is simply absent).

---

## 3. Against native ComfyUI

| native | ours | relationship | the difference that matters |
|---|---|---|---|
| `EmptyMiniMaxH3LatentAV` | absorbed into both conditioners | replaces | canvas and latent cannot disagree |
| `MiniMaxH3ImageToVideo` | `MiniMaxH3Conditioning` | replaces | refuses an empty prompt; a lone last frame anchors instead of being cover-cropped; canvas derived from the anchor; length snapped then ceiling-checked |
| `MiniMaxH3ReferenceToVideo` | `MiniMaxH3ReferenceConditioning` + three appends | replaces | sockets become an ordered tuple; suffix pairing becomes ownership; frame rate normalised from loader metadata; mono upmixed; soundtrack capped; latent grid read off the VAE rather than the pixels |
| `MiniMaxH3AddGuide` | none | — | arbitrary-frame guides are reachable only through core, and chain onto ours unchanged |
| `MiniMaxH3SigmaShift` | none | consumed | a core node, used as-is on every graph |
| `optimized_attention` / `_override` | `MiniMaxH3SageAttention` | replaces | per-module forward object patch |
| core `CLIPLoader` | `MiniMaxH3EncoderLoader` | wraps | core's own load, then refuses a silently incomplete load or wrong marker ids, and stamps what core's preprocessing will do (§5.1) |
| one `vae_dtype` | `MiniMaxH3VAEPrecision` | wraps | encode and decode split apart |
| `LoraLoaderModelOnly` | `MiniMaxH3PDDLoRA` | supplements | core's loader applies the backbone and silently skips the mechanisms that are not weight patches |
| width/height ints, image scaling | `Resolution`, `KeyframeCanvas`, `ReferenceFit`, `ReferenceVideoFit` | supplements | — |
| nothing | `SageChainAssert`, `Preflight`, `ProvenanceStamp`, `MarkerArm`, `SolAttnCurve` | no counterpart | — |

**The attention bypass is the sharpest divergence** (*read*). The native H3 path
honours `optimized_attention_override` fully — it passes `transformer_options`
down and boxes q/k/v as containers. Our object patch bypasses that entire layer,
and with it the global backend flags, the per-call override, the container
protocol and sage's own fallback. `attention.py::make_sage_override` exists
precisely to re-enter the layer the forward patch stepped around. Worth stating
plainly because it is easy to get backwards: `optimized_attention_for_device` is
**not** lost, because the H3 DiT never called it.

**The five reference contracts.** These belong to core's
`MiniMaxH3ReferenceToVideo`, and [`research/conditioning_nodes.md`](research/conditioning_nodes.md)
owns them. Core holds four. The fifth it holds only on the branch we take —
forced hooks drop the token tags where the merge path keeps them — and **ours
inherits that seam exactly.** It is the one contract our replacement does not
improve.

---

## 4. Against the vendor and the engines

Most of the chain matches. The divergences concentrate in three places, and
they are ranked here by whether they change the output or only the cost.

### 4.1 Changes the output

**1. The reference view the text encoder sees: three independent
implementations agree, and since 2026-09-13 so do our defaults.** sglang,
DiffSynth and diffusers all feed **one** prepared reference tensor, at a 2048
short edge with upscaling, to both the VAE and Qwen. `MiniMaxH3AppendRefImage`
now defaults to the same (`size_policy=max`, `dit_short_edge=2048`,
`allow_upscale=True`, `qwen_view=shared`; owner, `docs/wiki/decisions.md`).
From 2026-08-27 until then the node set a separate, much smaller 512 view for
the encoder, deliberate and dated: the fix for a two-speaker scene whose
dialogue was misattributed after upscaled references crowded the prompt out of
its own segment, *priced, not proven* on one render and one seed (CHANGELOG
0.82.0). That view is still available as `qwen_view=separate`. The ablation
that would judge it is built and unrendered: `bench/refview2_arms.json`, six
arms on five scene graphs (`h3_config.REFVIEW2_SCENES`). **This is the
strongest open candidate in this document.**

**2. `er_sde` against deterministic Euler.** sglang runs eta-0 Euler with no
noise after the initial draw. We run a stochastic multistep SDE that injects
fresh noise every step, and H3 declares no noise-scale override, so the term is
live. Deliberate and recorded in `workflows/h3_config.py::SAMPLING`; **enforced by
nothing** for the PDD arms, where `docs/checks.md` already carries the row.

**3. Video VAE precision — three-against-one on the shipped default, and
already graded.** diffusers pins the VAE's encoder, decoder and both conv layers
in fp32 and decodes under autocast; DiffSynth loads the release fp32 file. The
**three canonical graphs** load the fp16 build and wire no
`MiniMaxH3VAEPrecision`, so both halves are fp16 there.

**Corrected 2026-08-28, and the correction matters more than the finding.** An
earlier version of this row said the node "is simply unwired" and called wiring
it the cheapest thing here to test. Both halves were wrong. It is wired, in
`h3_probe_ref_vae_encoder_fp32*` against the matched `..._fp16*` control, and
[`comfyui_vendor_gaps.md`](comfyui_vendor_gaps.md) already names the pair. And
the knob is already answered: `bench/results/2026-08-21_vae_encoder_precision.json`
holds the graded comparison, which `bench/grade_vae_encoder_precision.py` opens
by explaining had to be a graded one — encoder precision is a numerical knob,
and a rendered pair cannot A/B one. **The defect was a dropped scope**: a source
read established that the three canonical graphs wire no such node, and this
document generalised it to "nowhere". Exactly the boundary error CLAUDE.md warns
about, committed inside a document arguing for care about boundaries.

What remains open is narrower and is a decision, not a measurement: whether the
shipped default should follow the graded result. That is the owner's call and
`comfyui_vendor_gaps.md` owns the row.

**4. Condition latents are the posterior mean, not a seeded sample.** Owned by
[`h3_references.md`](h3_references.md), open, unchanged.

**5. New — the condition-noise *draw* differs from sglang, though the recipe
matches** (*inference*). sglang draws in latent space at the target size and
prefix-slices before patchify; the ComfyUI path draws in row space after
patchify at the condition's own size. Two independent reasons the tensors differ
at one seed. `h3_references.md`'s table marks this row a match; on this reading
it should say "recipe yes, draw no". Proposed correction, not applied — that
table's owner is that file, and this is a static read.

**6. New — the audio stream rides the video schedule** (*inference*). ComfyUI
carries one schedule and scales the audio latent by a constant, converting the
velocity back inside the DiT; sglang, DiffSynth and diffusers all run two
independent schedules. The conversion is exact for a first-order update, but
`er_sde`'s higher-order corrections **and its injected noise** are built from
video-schedule quantities and applied to audio rows too. Nothing in this repo
covers it, and it is cheap to check at the call rather than at the output.

**7. The SLA router does not cover what the SLA LoRA was distilled on**
(*measured*, from the artifact header). The Turbo-SLA LoRA carries modules for
the fifty DiT blocks **and the token refiner**. The retired `MiniMaxH3SLARouter` patched the
DiT blocks only. `docs/open_experiments.md` #20 named this gap; the LoRA's own
key set is the confirmation it lacked. Two further asymmetries, both *read*: the
distillation pooled queries in larger blocks than our router does, and the
vendor kernel quantises q/k/v where ours is bf16 throughout — that operator is
not buildable on this box.

### 4.2 Changes only the cost

Sequence parallelism, AdaLN caching, CUDA graphs and step caching are all
sglang's and all absent here;
[`research/sglang_comparison.md`](research/sglang_comparison.md) owns that list
and prices it. LightX2V additionally has block prefetch, tensor and sequence
parallelism, and **block-granular text-encoder offload** — the last is the shape
of the fix for the host-memory incident during a bridge load, and we have no
equivalent.

Two apparent gaps are savings, not deficits (*read*): we omit sglang's padding
tail, whose rows sit in their own attention segment and cannot reach a live row,
and we resample reference audio once where sglang does it twice.

### 4.3 Where the agreement is worth banking

Character-for-character agreement across all four implementations on the
reference label rules, including the awkward case where a sounded video emits
its audio label immediately before its video label. No system prompt, no chat
template, the same encoder layer, no pooling, no truncation, the same tag
convention. The VAE normalisation statistics are bit-identical between ComfyUI
and DiffSynth (*measured*, parsed and compared for this document — the decimal
text differs, the doubles do not). H3's decode tiling is **not** a silent
ComfyUI branch: the H3 VAE hard-wires it and short-circuits ComfyUI's fallbacks,
matching the release. That last one corrects an impression
`sglang_comparison.md` leaves.

**Neither library is independent evidence about the seven markers.** Both
inherit the release tokenizer without touching the ids in code. Per CLAUDE.md
this question is settled; noting it here so the next reader does not mistake two
more implementations for two more votes.

---

## 5. More than one way to do it

Where the graphs could be wired two ways, this is what the choice costs.

### 5.1 Text-encoder loader

Every shipped graph wires `MiniMaxH3EncoderLoader` with the INT8 ConvRot
encoder (`h3_config.MODELS["clip"]`): core's own `CLIPLoader` load, then a
refusal if the load left parameters unpopulated or the marker ids are not the
release's, and a stamp saying what core's preprocessing will do
(`h3_encoder_loader.native_encoder_contract`, read out of core). The generator
refuses any file outside `workflows/h3_config.py::CORE_LOADED_ENCODERS`, so a
graph can never name a file the loader cannot open (*read*).

History. Until 2026-08-27 the graphs loaded a compressed-tensors W4A16 AWQ
artifact through `MiniMaxH3AWQEncoderLoader`, a format adapter that also
installed the artifact's own narrow still-image bounds; from 2026-08-29 they
loaded the INT8 file through core's bare `CLIPLoader`, which stamps nothing,
so `video_policy: "encoder"` resolved to `comfy` on every graph that asked for
it. On 2026-09-13 the adapter, its snapshots and the `encoder` policy were
deleted (`docs/wiki/decisions.md`). What is kept: a stock load path with the
guards core lacks, and the encoder that sits closest to the BF16 release at
the layer that matters. `REF_QWEN_SHORT_EDGE` is no longer a compensating
clamp on the default path; it pre-fills `qwen_view=separate`, and per §4.1 it
is a stated prior with one supporting render, not a measurement.

### 5.2 Attention

Four live configurations: sage+Sol (the default), sage alone, the SLA router,
and stock SDPA. `bench/check_attention_defaults.py` is the strongest check in
the repo — it grades by reachability from the output node rather than node
presence, pins values rather than presence, derives its exempt classes instead
of listing them, and asserts each exemption is *necessary*. It runs green on
this tree (*measured*, this session).

### 5.3 Step reduction

Three loaders that are **silently non-interchangeable** (*read*). The turbo-pack
file cannot be applied by the stock loader at all. The stock loader on a PDD
file would apply the backbone and skip the two mechanisms that are not weight
patches — producing a render that is undistilled and looks fine. The deeper fork
is schedule ownership: PDD graphs no longer carry a `BasicScheduler` at all.
See the PDD document.

### 5.4 Conditioning, resolution, VAE, caching

Our two conditioners replace core's everywhere. The reference `image_policy` is
`comfy` on every reference graph; `video_policy` is `comfy` on all but the
`release` probe arm (walk `h3_config.graph_paths`). The `encoder` value most
graphs used to select never ran and was removed on 2026-09-13 (§2).
`MiniMaxH3Resolution` is convenience over typing two integers, and is wired
almost everywhere. The VAE-precision and cache axes are single-graph probes,
both carrying committed measurements. A conditioning-encode cache, if one is
ever built, should hold the joint encode only: a peer pack that cached
separately encoded sections reverted it after reported generation distortion
(`coderef/ComfyUI-UtilsCollection` `d1921ae`; [`wiki/references.md`](wiki/references.md)
describes the pack).

**Output is not an axis.** Every graph wires the same video writer with every
input constant except the filename prefix.

---

## 6. What is enforced by nothing

New rows, or rows this pass sharpened. `docs/checks.md` holds the standing
audit; these are candidates for it, not additions to it.

| requirement | status |
|---|---|
| Sol's live-module surgery is undone when the node is bypassed | **nothing.** It rebinds module attributes and registers hooks outside the patcher, keyed on object identity, and nothing removes them |
| a PDD arm is consumed by `euler` | **nothing** — already in `docs/checks.md`, re-verified here |
| `MiniMaxH3Preflight` stays wired in shipped graphs (`SageChainAssert` left them 2026-09-17) | **nothing** |
| the two attention nodes' order is what the graph actually contains | `check_attention_defaults.py`, statically (reachability, the dense floor); the call-time `SageChainAssert` left the generated graphs 2026-09-17 |
| the SLA router covers the modules the SLA LoRA adapts | **nothing**, and it currently does not (§4.1) |
| the sigma tail lands outside Sol's window | **nothing.** The node logs the window and tells the reader to check; it never sees the sigmas |
| the audio stream's higher-order sampler terms are valid on the video schedule | **nothing** (§4.1) |
| `qkv_proj` row order matches the checkpoint's convention | **nothing on any side.** The same key name denotes two different row orders across implementations; our converter targets the right one on the strength of a comment |
| `_SPANS` in `sol_attn_h3.py` stays bounded over a long-lived server | **nothing.** Keyed on `id(position_ids)`, one entry per distinct packed layout, each keeping its layout alive on purpose so the id cannot be recycled, and nothing evicts an entry. How fast it grows across many distinct shapes is unmeasured; flagged 2026-09-10, not chased |

One methodological item, which is not a check but belongs on the record:
**noise for the video and audio streams comes from one seeded generator consumed
in order**, so the audio noise depends on the video latent's element count. A
matched-seed A/B whose arms differ in canvas or length is **not** matched on the
audio stream. [`eval_comparison.md`](eval_comparison.md) owns the A/B process
and does not carry this caveat.

---

## 7. Actionable, cheapest first

1. **Fix the stale generator comment.** `workflows/build_workflows.py` tells a
   reader one turbo arm ships "sage on and Sol absent", giving reasoning the
   same file explicitly retracts a hundred lines later; the emitted graph
   carries live Sol (*measured*, this session). The configuration is right and
   the A/B pairing is intact — only the prose is inverted. No check reads
   generator comments.
2. **Decide the shipped VAE default against the result we already have.** Not
   a measurement — `bench/results/2026-08-21_vae_encoder_precision.json` graded
   it on 2026-08-21 and the probe arms exist. The open part is whether the
   canonical graphs should follow it, which is a decision.
3. **Extend the SLA router to the token refiner**, or record the asymmetry
   against `open_experiments.md` #20 now that the LoRA's key set confirms it.
4. **Run the encoder-bounds arm.** Weights fixed, bounds varied — the arm
   `h3_config.py` already describes and which is not reachable from a graph.
   It is what turns §4.1's first item from priced into proven.
5. **Propose the two `h3_references.md` corrections** to that file's owner: the
   condition-noise draw, and the audio-schedule question.

---

## 8. What this pass did not examine

Static reads only. Nothing here was rendered, no GPU was used, and no claim
below the level of "the code says" was established except where marked
*measured*. Specifically not examined: whether any divergence named here is
visible in output; the single-frame path, which is parked; the audio-refine and
looping packs; and the bench harnesses except where a check was cited.
