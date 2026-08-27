# Reference node redesign: fold the fit, and move the ceiling to the conditioner

**Status:** Implemented
**Date:** 2026-08-24
**Scope, as committed:** `reference_geometry.py` (new),
`reference_conditioning.py`, `reference_fit.py`, `h3_awq_encoder.py`,
`workflows/build_workflows.py` and every regenerated graph; the static
readers `bench/preflight_graph.py` and
`bench/audit_shipped_reference_bounds.py`; the checks
`bench/check_reference_runtime.py`, `bench/check_generator_constants.py`,
`bench/check_reference_fit.py`, `bench/check_typed_reference_consumers.py`,
`bench/node_id_manifest.json`, and the deletion of
`bench/check_short_edge_override.py`; and the docs that stated where these
knobs live -- `docs/h3_references.md`, `docs/comfyui_vendor_gaps.md`,
`docs/research/conditioning_nodes.md`, `docs/h3_conditioning_end_to_end.md`,
`docs/research/sglang_comparison.md`, `docs/checks.md`.

One image reference cost four nodes: `LoadImage` -> `MiniMaxH3ReferenceFit` ->
`MiniMaxH3AppendRefImage` -> `MiniMaxH3ReferenceConditioning`. It now costs
three. That was the smaller half of the change. The larger half is that the
Qwen still-image ceiling moved to the conditioner, where the CLIP is in scope
and the answer is knowable, and became `image_policy`.

## What shipped

### 1. `image_policy` on `MiniMaxH3ReferenceConditioning`

The sibling of `video_policy`, with the same three arms and the same shape:
`comfy` (default, passthrough — exactly what every graph got before), `release`,
`encoder`. `_qwen_image_settings` selects bounds and geometry per policy, and
`_configured_qwen_image_size` pre-applies the selected policy's `smart_resize`
*before* the VAE encode, so both towers encode one tensor at one size.

Why it had to move. One ceiling has three live values:

| policy | bounds | source |
|---|---|---|
| `comfy` | 3,136 – 12,845,056 | `process_qwen2vl_images` signature defaults |
| `encoder` | 200,704 – 301,056 | the loaded artifact's `processor_config.json` |
| `release` | 65,536 – 16,777,216 | the release's `preprocessor_config.json` |

`reference_fit.py::qwen_max_pixels()` read the first by introspection and
applied it universally. That is right for a native BF16 graph and wrong by
orders of magnitude under the AWQ adapter — and the fit node has no `clip`, so
it cannot tell which it is feeding. `h3_awq_encoder.source_image_pixel_bounds()`
names the defect in its own docstring, added in `329ab25` earlier the same day
as this change.

**The design that got this wrong assumed it was a separate, later change.** It
is not: the pattern was already implemented next door for video, and the
override plumbing already existed too — `install_source_processors(clip,
image_bounds=...)` takes an override, records `_h3_image_bounds`, and logs it.
Only the selector was missing, plus one accessor —
`source_image_patch_geometry()`, a mirror of its video sibling reading the same
config block `source_image_pixel_bounds()` already read.

Measured, on this box, for one reference prepared at the release's 2048 short
edge:

| source | `comfy` | `release` | `encoder` |
|---|---|---|---|
| 3648x2048 | unchanged | 3648x2048 | **704x384** |
| 2048x2048 | unchanged | 2048x2048 | 544x544 |
| 224x224 | unchanged | 256x256 | 448x448 |

The first row is the current artifact's ceiling costing roughly 25x the visual
detail of a reference sized the way the release serves it — independently
reproducing what
[`still_policy_token_cost.md`](qwen3-vl-special-tokens-post-training/canonical/2026-08-24_still_policy_token_cost.md)
measured. The third row is the **floor**, which the retired `keep_towers_matched`
never modelled at all: it clamped a ceiling and had no opinion about a
reference too small for the declared policy.

### 2. The fit folded into the append

`MiniMaxH3AppendRefImage` gained `allow_upscale` and `short_edge`, appended
after `references`. `MiniMaxH3ReferenceConditioning` performs exactly one
resize, with the canvas in scope.

**This differs from the original design, which had the append do the resize.**
It cannot: `size_policy='match'` sizes from the target canvas area, and the
append has no canvas. Recording the decision on the record and resizing once at
the compiler works for both policies, and eliminates the redundant resample and
the redundant quantization outright rather than guarding a second resize site.

Consequently the proposed `image` and `latent_rows` outputs on the append were
**not** added — with the resize at the conditioner the append cannot honestly
produce a fitted tensor. Nothing consumed the fit node's outputs on any graph,
so this forecloses nothing, and appending outputs later is permitted.

### 3. Retired, but not removed

`lift_downstream_clamp` and `keep_towers_matched` stay on `MiniMaxH3ReferenceFit`
as inert inputs with tooltips saying so. This followed the `vendor_tokens`
precedent on the conditioner, and **that precedent was reversed on 2026-08-27**:
the owner ruled saved-graph compatibility not worth keeping ("it's just me
using this"), `vendor_tokens` was removed from both conditioners, and
`size_policy`'s dead siblings moved inside a `DynamicCombo` branch rather than
staying visible and inert. The reasoning below -- that an inert input costs a
tooltip and a removed one breaks saved graphs -- held only while the second
half was a cost anyone was paying. These two survive because
`MiniMaxH3ReferenceFit` is itself retired from the shipped path, not because
the argument still stands.

The machinery behind them is gone: `_downstream_ref_image_size`,
`arm_short_edge_override` / `disarm_short_edge_override`, the
`MiniMaxH3ReferenceToVideo` wrapper, and `fingerprint_inputs` — which returned
`float("nan")` whenever `lift_downstream_clamp` was set, so arming a feature
that could never work also permanently disabled that node's cache.

`bench/check_short_edge_override.py` retired with the code it guarded. Worth
recording why it was green: its fixture supplied synthetic prompts containing
`MiniMaxH3ReferenceToVideo`, so its input pre-satisfied the outcome and it
could never notice that no shipped graph contains that node.

### 4. One sizing implementation

`reference_geometry.py` owns both stages: `fit_reference_image` for role
sizing, and `qwen_image_settings` / `qwen_image_size` for the still policy,
which moved there so the static readers can reach them without importing the
node package. Its callers are `MiniMaxH3ReferenceConditioning` (which performs
the resize), `MiniMaxH3ReferenceFit` (legacy), `bench/count_packed_rows.py`,
and the two static readers `bench/preflight_graph.py` and
`bench/audit_shipped_reference_bounds.py`. The post-training calibration
builder is meant to be the sixth.

**`MiniMaxH3AppendRefImage` is deliberately not among them.** It records the
decision on the record and validates it; it never sizes anything, because the
canvas `match` needs is not in its scope. An earlier draft of this document
listed it as a caller, which was wrong.

`bench/preflight_graph.py` was the last copy and was the one that mattered: it
exists to price what you are about to render, so a copy that disagrees with the
node reports a sequence length nobody will get. It carried its own `fit()` plus
an inline scale selection until this change.

That is not tidiness. The active plan names two strata that are exactly this
function's arguments — a primary `max` with upscaling off, and a separately
named 2048-short-edge upscale-allowed stress stratum — and requires every row
to record which one it came from. Two implementations of that arithmetic is
drift nothing would catch, because both copies would be individually correct
and would disagree only on inputs neither author tried.

### 5. `match` ignores `allow_upscale`, on purpose

Found while auditing this document against the committed code, not before it.
The first implementation applied `allow_upscale` to both policies, while the
append node warned that `match` does not read it. One of the two was lying, and
it was the code: before the fold this branch had no upscale knob to read at
all, so honouring it would have silently changed every saved graph carrying
`size_policy='match'`. Core also clamps with `min(1.0, ...)` in **both** its
modes. `match` now clamps unconditionally and the warning is true. Verified
against the pre-fold arithmetic across four sources and both flag states.

## Corrections to the original design

**`MiniMaxH3ReferenceVideoFit` is not dead and was not deleted.** The design
listed it as a dead path on the grounds that zero graphs wire it. Its module
docstring says why that is deliberate: it is a *reporter* for native-core
paths, explicitly distinguished from `video_policy=release`, and reporting is
its deliverable. By the "wired by zero graphs" test every diagnostic node in
the repo is dead. `qwen_max_pixels` and `clamp_to_qwen_ceiling` therefore stay
exported for it — and on a native-core path Comfy's default genuinely **is**
the right ceiling, which is the whole reason the same helper was wrong in the
fit node and right here.

**The static-analysis argument was overstated.** The design credited the fold
with fixing `bench/audit_shipped_reference_bounds.py::_trace_to_loader`'s eight-hop
walk. The walk was eight hops only because the fit node sat between the append
and the loader; after the fold the append links straight to `LoadImage`. The
loop is retained for saved graphs that still wire a fit node, and now composes
the two rather than taking the first it finds.

## What a review caught afterwards

The session that shipped this is written up at
`internal/postmortems/2026-08-24_session_reference-fold-and-image-policy.md`
(gitignored, local). Bare path rather than a link, matching how
`docs/evidence.md` and `docs/roadmap.md` cite that directory.


An `xhigh` code review of the committed change found fifteen issues, several
reproduced by running the code rather than inferred. They are worth recording
because most are not in the design at all -- they are the difference between a
design being right and a change being right.

**Three were defects introduced by the change.**

- `bench/count_packed_rows.py` imported the sizing helper that moved into
  `reference_geometry`, so it died with `ImportError` on every invocation. The
  dangling-symbol sweep before committing covered the retired override
  machinery and missed the helper that moved.
- The identity guard returned `image[:1]`, dropping the `[..., :3]` channel
  slice that `_resize` performs on every other path. A four-channel RGBA
  reference already at its target size reached `vae.encode` with four channels.
- Two preparation stages were merged rather than composed. A legacy fit node
  upstream of the append had its arguments combined by taking the larger
  `short_edge` and OR-ing the upscale flags; they compose in order. The merge
  over-priced by the square of the ratio whenever the append was narrower --
  `Fit(2048, upscale) -> Append(1024)` really yields 1024x1024 and was reported
  as 2048x2048. **This one was flagged as unvalidated when the review was
  commissioned** and the answer came back that it was wrong, which is the
  cheapest way this list was produced.

**The rest were gaps rather than breakage**, and the two that matter:

- `bench/preflight_graph.py` never read `image_policy`, so under `encoder` it
  reported 4,096 rows for a reference the DiT receives at 289. The tool exists
  to price the sequence, so a stage-two-blind price is the failure it is for.
  Its `_vision_bound_warnings` also asserted the opposite of what `release` and
  `encoder` do, and now fires only under `comfy`.
- **Nothing asserted that the folded knobs reach the compiler.** Replacing
  `record.short_edge` / `record.allow_upscale` with the function defaults left
  `check_node_ids`, `check_reference_fit` and the two `image_policy` contracts
  green while every shipped graph silently lost its upscale -- a 4x change in
  sequence length. The knob with the largest blast radius in the change had no
  runtime control. `append_sizing_reaches_the_encoded_geometry` is that control,
  asserted against the latent grid the DiT is handed and shown to go red under
  exactly that mutation.

The pattern in all five: the design was about where a decision should live, and
every one of these is about whether the decision is actually read at the far
end. A design review cannot produce this list; only running the code can.

## What was rejected

**An Autogrow "star"** — `reference_0 .. reference_11` on the conditioner,
replacing the chain. Verified buildable against installed ComfyUI. Rejected on
saved-graph compatibility: the conditioner's single `references` link would
become `reference_0..11`, so every saved graph wiring `"references": [id, 0]`
stops validating, including the owner's graphs outside this repo, which no
check can see.

**Folding `LoadImage` into the append.** Every reference image is loaded from a
file today and no `LoadImage` in any graph has more than one consumer, so the
1:1 evidence is identical to the fit's. Rejected anyway, on three counts: it is
an input *removal*, the shape `check_node_ids.py` exists to prevent and the same
ground the star was rejected on; it takes ownership of a core node that recently
moved to `InputImpl.VideoFromFile` and carries an `IS_CHANGED` file-hash cache
contract, so missing it serves a stale tensor silently; and it forecloses
non-file sources — `workflows/image/` produces single frames in this repo, and
the socket already permits chaining them.

If the source-identity half is wanted later (path plus content hash on the
record, which the plan's media hashing would use), it should arrive as an
additive `MiniMaxH3LoadReferenceImage` beside the socket-taking node, never
instead of it. Core does the same with `LoadImage` / `LoadImageOutput` /
`LoadImageMask`.

## What is still not fixed

`image_policy` defaults to `comfy`, which means the shipped graphs still get
core's behaviour: the still is handed to the text encoder as core hands it, and
whatever processor the loaded CLIP carries resizes it afterwards — for Qwen
alone, after the VAE has already encoded. Choosing a different default is a
behaviour change to every reference graph and wants a measurement, not a
decision made while folding nodes.

## Controls

`bench/check_reference_runtime.py` gained two contracts, and both were shown to
fail before being trusted. Three deliberate violations — collapsing the policy
selector to one branch, removing the floor so only ceilings clamp, and letting
`comfy` borrow the release processor rather than refusing — each turn them red,
and the unmutated pair is green.

`bench/check_generator_constants.py` now reads `short_edge` from the append
node. It previously read it from the fit node, and when the graphs stopped
carrying that node the set comprehension simply went empty; an empty set
compares unequal to every value, so its three cases went red for the right
reason by accident. It asserts against the empty case explicitly now.

`bench/check_node_ids.py` classified every schema change as an append and
nothing as a removal or a reorder, which is the compatibility claim above
stated by the guard rather than by the author.

Old and new graph shapes were driven through the sizing function across six
sources and both upscale states, and agree everywhere: a saved
`LoadImage -> Fit -> Append` graph and a folded `LoadImage -> Append` graph
produce identical geometry, because the append's `max` fit is idempotent on an
already-fitted reference. `match` was checked separately, after the defect in
section 5 was found, against the pre-fold arithmetic.

Routing `bench/preflight_graph.py` through the shared function reproduced its
previous prices exactly on both arms of the shipped reference graph -- which is
what makes it a refactor rather than a behaviour change.

## What stays as it is

The chain itself, the per-reference record model, `reference_order.py`'s label
assignment and ownership validation, and the conditioner's schema beyond the
appended input. Per-reference sizing stays per-reference and is **not** hoisted
onto the conditioner as one global knob — each record carries its own policy,
geometry, latent grid and rotary slot, and mixed geometry is supported by
construction. `image_policy` is deliberately the opposite kind of knob: it
selects whose *processor* applies, which is a property of the loaded checkpoint
and cannot vary per reference.
