# Where ComfyUI's H3 path differs from the vendor's

last updated: 2026-08-25

Every known divergence between this ComfyUI install and the MiniMax H3 release,
in one place, with what each one costs a working user.

**This file is a snapshot, not an authority.** Every fact in it is owned by
another document, named in each section. It exists because the ownership rule
that keeps those documents from drifting also meant that answering "what are all
the gaps" required knowing which of three files to open. **If this file
disagrees with an owner, the owner is right** and this one is stale. Regenerate
it by re-reading the owners rather than by editing it in place.

**Native status and local handling are separate facts.** In this document,
"native ComfyUI" means code provided by the installed ComfyUI checkout, without
this custom-node pack. "Local handling" means a node, check, preflight rule or
shipped workflow in this repository. A local workaround can make this repo's
graphs safe without resolving the ComfyUI/vendor divergence for anyone else;
it is therefore described as **locally handled**, never as a closed native gap.
Conversely, a native ComfyUI fix remains a native fix after this repo retires
its workaround and keeps only inert saved-graph compatibility slots.

Sources, and what each is worth:

- **Measured here** — a record in `bench/results/`, named where it applies.
- **Read from source** — `coderef/sglang` at `a7ec6b97f7` and `coderef/diffusers`,
  against ComfyUI's own files. A source read is an inference, not a run.
- The 2026-08-21 reads were made at `coderef/sglang` `a41da991c8`. Every cited
  line still resolves; the surrounding prose has not been re-read.

---

## The two kinds, because they are fixed differently

**Config inheritance.** The release ships a value in a config file. sglang reads
it through `AutoTokenizer` / `AutoProcessor` from the paths `model_index.json`
names
(`coderef/sglang/python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py:70`).
ComfyUI hardcodes its own. These are cheap: the fix is to read the file.

**Behavioural.** sglang decided something we did not. Copying it is a design
choice, not a correction.

---

## Adjacent checkpoint-format status — AWQ is not one loader format

This is not a MiniMax-release divergence, but it belongs beside the gap table
because the filenames otherwise make the wrong conclusion easy.

Core ComfyUI **does natively support the shipped NVFP4-AWQ H3 encoder** on
compatible hardware. Its file uses the H3 namespace core detects
(`visual.*`, `model.layers.0` through `49`) and carries per-layer
`comfy_quant` records: 350 language linears declare `format=nvfp4`. The stock
`CLIPLoader` is the correct owner of that artifact. This is native ComfyUI
support, not code from this repository.

The canonical graph selection, `qwen3vl_32b_minimax_h3_w4a16_awq.safetensors`, is AWQ-calibrated
but is a different representation: compressed-tensors W4A16, packed `int32`
weights, group-128 scales, and the full Hugging Face
`model.language_model.*` namespace. Core lists it because the file is in
`models/text_encoders`, then detects that namespace as Qwen3-VL-8B and builds
width 4096 for width-5120 tensors. The resulting load failure was reproduced
on 2026-08-23. `MiniMaxH3AWQEncoderLoader` is this repo's local adapter for
that representation: it accepts any selected filename only after its embedded
metadata and complete adapted tensor inventory satisfy the versioned contract.
It does not supply or imply generic AWQ support in core.
[`bench/check_h3_awq_encoder.py`](../bench/check_h3_awq_encoder.py) controls
both sides from the real files. See
[`h3_awq_encoder.md`](h3_awq_encoder.md) for the full responsibility boundary,
packing adaptation, processor behavior and execution-path comparison.

---

## Summary

Priority is by what it costs a working user, not by how interesting it is.

| # | gap | kind | native ComfyUI status | handling in this repo |
|---|---|---|---|---|
| 1 | Seven special tokens absent from the tokenizer | config | **fixed in the installed checkout by merged PR 15808** | local fallback retired; native behavior is required and audited |
| 2 | Reference video frame rate assumed, not enforced | behavioural | open | typed nodes normalize from owned loader metadata; shipped graphs also retain and check `force_rate=24` |
| 3 | Reference image floor (`min_pixels`) | config | open | preflight reports the divergence; no general runtime parity implementation |
| 4 | Reference image ceiling (`max_pixels`) | config | open | custom fit nodes can opt in to keeping VAE and Qwen sizes matched; core remains unchanged |
| 5 | Reference soundtracks not truncated | behavioural | open | all shipped graphs now use typed internal caps; native socket graphs remain exposed unless they trim upstream |
| 6 | Reference media never upscaled, and never reported | behavioural | sizing divergence remains; native path does not report the choice | fit nodes report it; the typed conditioner has an opt-in atomic release-video policy, while shipped defaults remain native-compatible |
| 7 | Mono reference audio raises | behavioural | open | typed nodes upmix mono and refuse ambiguous multichannel input; legacy preflight reports it |
| 8 | VAE encode precision, and mean vs sample | behavioural | open | measured only; no claimed fix |
| 9 | H3 VAE tiling as a runtime branch | behavioural | **not a gap in the installed native implementation** | documented as fixed H3-owned policy; no custom-node fix claimed |
| 10-13 | Partition gate, AdaLN cache, CUDA graphs, step caching | behavioural | architectural differences | researched or explicitly declined; no native-equivalence claim |

---

## 1. Seven special tokens — fixed natively in installed ComfyUI

Owner: [`research/official_weights_metadata.md`](research/official_weights_metadata.md).

The release declares twenty `additional_special_tokens`. ComfyUI's bundled
`qwen25_tokenizer` directory declares thirteen. The seven missing are `<d>`,
`</d>`, `<|cutoff|>`, `<|lyrics_start|>`, `<|lyrics_end|>`,
`<|caption_start|>`, `<|caption_end|>`.

**Practical impact: every dialogue prompt**, because the release's prompt guide
mandates `<d>` around all spoken lines. And the cost is not confined to the
marker. BPE has no reason to stop at the angle bracket, so the fragments fuse
with the text on either side:

| | release emits | ComfyUI emitted |
|---|---|---|
| `<d>[English]` | `<d>` `[` `English` | `·<` `d` `>[` `English` |
| `window.<\|cutoff\|>` | `window` `.` `<\|cutoff\|>` | `window` `.<` `\|` `c` `utoff` |
| `<\|lyrics_start\|><d>` | two tokens | six fragments, fusing into `><` |

A full stop before `<|cutoff|>` is dragged into `.<`, so a marker retokenizes
the sentence *before* it. Measured 2026-08-22 across nine prompt shapes
([`bench/results/2026-08-24_h3_marker_tokenization_native.json`](../bench/results/2026-08-24_h3_marker_tokenization_native.json)): in an ordinary
two-person dialogue prompt, 9 of 91 non-marker tokens come out different.

**Native ComfyUI status: fixed.**
[Comfy-Org/ComfyUI PR 15808](https://github.com/Comfy-Org/ComfyUI/pull/15808)
(kijai) is merged, and this installed checkout contains it as commit
`924743af`. It declares the seven on a `Qwen3VLSDTokenizer` subclass so every
consumer gets them, including core's `MiniMaxH3ReferenceToVideo`. An older
ComfyUI install without that commit still has the defect described below.

Verified here against the release tokenizer: all nine scenes reproduce its ids
exactly, the reference path carries the marker beside its vision blocks, and a
marker-free prompt is byte-identical to the reconstructed legacy arm.
[`bench/audit_h3_marker_tokenization.py`](../bench/audit_h3_marker_tokenization.py)
is the harness and refuses an install that lacks the native fix.

**Handling in this repo:** no tokenizer patching remains. The conditioning
nodes rely on native ComfyUI. Their old `vendor_tokens` inputs and the
standalone `MiniMaxH3VendorTokens` node are inert compatibility tombstones so
saved UI graphs retain their positional widget contract.

### Do the seven tokens touch Qwen3-VL's vision tower?

**No.** The question is a fair one because the release's `processor/` directory
carries the tokenizer config, so the tokens do sit next to the vision
configuration. They do not reach the tower:

- The tower consumes pixel patches. No vocabulary reaches it.
- The vision sentinels are hardcoded ids in `comfy/text_encoders/minimax.py:29-30`,
  never a vocabulary lookup.
- The seven append *above* the highest existing added token, so no id shifts.

The assertion that closes it: over two images, an odd-frame video and an audio
reference, a **marker-free prompt tokenizes byte-identically with and without
the fix, vision structure included**. Had any vision id moved, that arm fails.

Why the ids cannot move, and why the tower has no vocabulary to align in the
first place (the seven are appended above every existing id; native ComfyUI
splices vision features by list position and never materialises
`<|image_pad|>`), is walked through the installed code in
[`research/official_weights_metadata.md`](research/official_weights_metadata.md),
"How the fixed install realises them".

**The useful reframe.** `processor/` bundles two different things, because HF's
`AutoProcessor` wraps both halves: a tokenizer config, and the image and video
preprocessor configs. PR 15808 fixes the text half. Gaps 3 and 4 below are the
vision half, and they are untouched. **Those are the ones that change what Qwen
actually sees.**

---

## 2. Reference video frame rate — assumed, not enforced

Owner: [`h3_references.md`](h3_references.md).

Both implementations target 24 fps. Only one enforces it.

sglang applies it as an ffmpeg `fps=` filter in the same decode pass that does
rotation, Lanczos scaling and square-pixel normalisation
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:397`),
so a 30 fps source becomes constant-rate 24 before anything sees it. ComfyUI
applies no filter: `FPS = 24` at `comfy_extras/nodes_minimax_h3.py:30`, no node
in that file exposes an fps input, and it assumes whatever the loader hands it
is already 24.

**Practical impact: any reference video that is not 24 fps.** The 2 fps
conditioner subsample steps by `FPS // 2` and stamps timestamps at `i / 2.0`.
Both are right at 24 and wrong at anything else. Measured on three 6.00-second
clips differing only in frame rate:

| source | `force_rate` | H3 reads it as | error | last label |
|---|---|---|---|---|
| 24 fps | 0 or 24 | 5.875s | 0.0% | `<5.2 seconds>` |
| 25 fps | 0 | 5.875s | +4.2% | `<5.2 seconds>` |
| 30 fps | 0 | **7.292s** | **+25.0%** | `<7.0 seconds>` |
| any | 24 | correct | 0.0% | correct |

At 30 fps the model is told a six-second reference holds seven and a quarter
seconds of action, and the spoken-word timestamps stretch with it.

**Workaround: `force_rate=24` on the loader.** A 24 fps source is unaffected
either way, which is exactly why testing on one proves nothing.
[`bench/check_ref_prompt_labels.py`](../bench/check_ref_prompt_labels.py) fails the build if any loader feeding a
reference socket drops off 24.

**Native ComfyUI status: open. Handling in this repo:**
`MiniMaxH3AppendRefVideo` requires the frames' own `VHS_VIDEOINFO`, records its
`loaded_fps`, and `MiniMaxH3ReferenceConditioning` normalizes the frames to 24
fps before either Qwen or the video VAE sees them. Every shipped reference
workflow now uses that local path and also retains `force_rate=24` so migration
did not change media policy. The native socket node still has no metadata input.

Two design consequences of sglang doing this in one ffmpeg pass: **fps never
enters its API surface** — the caller asks for `duration_seconds` and the frame
count is derived
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py:291-295`)
— and the decoded array is shared by Qwen and the VAE by construction.

The 2 fps subsample itself, including the index pad and merged-pair timestamp,
is **confirmed identical** to sglang. The rate is the whole gap.

**And the per-pair vision call is now tested, not argued.** ComfyUI presents a
reference video to Qwen3-VL as separate two-frame calls at `grid_thw =
[1, H, W]` where LightX and sglang pass the whole sampled clip at `[T, H, W]`.
`internal/codex/2026-08-21_h3-conditioning-qwen-independent-review.md` section 2
calls this "the important non-gap" and argues structural equivalence -- then
says in its own words that it "should be tensor-tested, but it is not". It now
is: [`bench/grade_video_block_presentation.py`](../bench/grade_video_block_presentation.py)
runs both presentations through the real `Qwen3VLVisionModel` at the
`qwen3vl_32b` vision config and the merged and deepstack outputs agree to
7.5e-06, inside fp32 reassociation noise. **Two controls establish that the
comparison can see a difference** -- reordering the pairs moves it by 7.0, and
perturbing one pair by 1e-3 moves it by 6.0e-03 -- and the harness exits
non-zero rather than reporting a pass if either fails to separate. The
mechanism is `comfy/text_encoders/qwen35.py:660-663`: `cu_seqlens` splits attention at every
`h*w`, so attention never crosses a frame in either presentation, and the
position construction carries no temporal term.

**Random weights are the right instrument and not a shortcut**: the claim is
about how the tower routes tokens, which is architecture rather than anything
learned. It tests nothing weight-dependent, and nothing in the claim is.

**The Qwen tower gets a SECOND, duration-aware resize the VAE never sees.**
Established 2026-08-22 by executing the release processor, after this section
claimed the opposite for several hours and was refuted by codex tracing the
call graph. The wrong claim rested on three sglang docstrings saying the
decoded array is "shared by Qwen and the visual VAE"
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:382`).
That is true and it is the *source* array. One call later,
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py:488` sends the 2 fps sampled view
through `proc.video_processor(...)` -- the release `Qwen3VLVideoProcessor` --
and the VAE consumes the shared array directly. **Shared source is not
identical tower resolution**, and reading docstrings instead of following the
call is what produced the error.

Its budget is duration-aware: `vendor_config/video_preprocessor_config.json`
sets `size.longest_edge` to 25,165,824 px **across the clip**, so more sampled
frames means each is smaller. Run against the real processor at 31 sampled
frames (362 target frames at 2 fps, padded to 32, 16 temporal blocks):

| fed to the processor | `grid_thw` | effective | Qwen rows |
|---|---|---|---|
| 1344x768 | `[16, 42, 74]` | 1184x672 | **12,432** |
| 960x544 (our source) | `[16, 34, 60]` | 960x544, untouched | **8,160** |

**At our source size the duration-aware pass is inactive**, and that is the
part neither the report nor the refutation had: 960x544 over 32 frames is
16.7M px, under the 25.2M budget, so the processor returns the frames
unresized and ComfyUI's per-pair path lands on the identical grid. The
divergence for this clip is **entirely gap 6 below** -- sglang upscales the
reference video to 1344x768 first, and the duration-aware pass then pulls it
back to 1184x672.

**It has a threshold, and every H3 length below 311 frames sits under it.**
Swept against the real processor rather than derived -- a first attempt
derived it from `budget // pixels_per_frame` and was wrong twice, on the frame
count and on the mapping; codex caught both. The executed boundary at
1344x768:

| sampled frames | `grid_thw` | effective |
|---|---|---|
| 24 | `[12, 48, 84]` | unchanged |
| 25 | `[13, 48, 84]` | unchanged -- 13 blocks, but `smart_resize` ties to even and budgets it as 24 |
| **26** | `[13, 46, 80]` | **1280x736, first resize** |

Sampling is `ceil(N / 12)`, so 26 samples starts at **301 target frames**, and
the first legal `17n+5` at or above it is **311**. At our un-upscaled 960x544
the first resize is at 50 sampled frames -- target 589, first legal 600 --
which is outside the legal range entirely, so **the pass can never activate on
this clip as ComfyUI feeds it.**

So the coupling below is length-dependent, and the boundary is exact:

- **124 through 294 frames**: only the upscale differs between the two paths.
- **311 through 362 frames**: the upscale and the duration-aware limit both
  apply.

**Which makes the duration-aware pass a cost LIMITER, and couples the two
gaps.** Closing gap 6 alone -- upscaling to 1344x768 and feeding Qwen per pair
as now -- gives 84x48/4 = 1,008 tok/block x 16 = 16,128 rows, **3,696 more
than the vendor's 12,432**. So the correct order is both or neither: an
upscale without the duration-aware resize overshoots the release rather than
matching it.

**None of this touches the per-pair result above.**
`grade_video_block_presentation.py` held the grid fixed and asked whether the
*call shape* diverges. It does not. This is a different question -- what grid
the tower is handed -- and the answer is that it can diverge, through the
upscale.

---

## 2b. Token tags vanish under forced CLIP scheduling

Owner: [`research/conditioning_nodes.md`](research/conditioning_nodes.md).

`CLIP.encode_from_tokens_scheduled` has two branches. The one shipped graphs
take asks `encode_from_tokens(..., return_dict=True)` and merges the encoder's
third return value, so `minimax_token_tags` reaches conditioning. **The
forced-hook branch does not**: `comfy/sd.py:379-381` reads `o[:2]` and builds
its `pooled_dict` from scratch, so the tags are dropped and the DiT tags every
row as text.

**Measured 2026-08-22, not read**: driven with a stub patcher carrying
`forced_hooks`, the conditioning extras come back as `clip_start_percent`,
`clip_end_percent`, `pooled_output` and nothing else. Found by codex reviewing
the seam rather than by anything here noticing.

*Impact: none on this repo today* -- no shipped graph wires a CLIP hook, so
the branch is unreachable from `workflows/`. It is recorded because the
condition is silent, and because anyone wiring hooks onto an H3 graph would
get a quietly worse render with no error.

**Recorded, not fixed, and not failed on.**
`bench/check_reference_contracts.py`'s case 5c asserts the CURRENT state --
green while core still drops them. If it flips, upstream fixed it and the case
and this entry should be retired rather than repaired, the same retirement
contract as gap 7's mono gate.

---

## 3 and 4. The image preprocessor bounds

Owner: [`h3_references.md`](h3_references.md).

ComfyUI leaves `min_pixels` / `max_pixels` on the shared Qwen2-VL helper's
signature defaults and applies the same pair to stills and video blocks alike.
The release ships different values, and different ones for each:

| bound | H3 release | ComfyUI |
|---|---|---|
| image min pixels | 65,536 | 3,136 |
| image max pixels | 16,777,216 | 12,845,056 |
| video min pixels | 4,096 | 3,136 |
| video max pixels | 25,165,824 | 12,845,056 |

The patch **geometry** is right on both sides — `patch_size=16`,
`temporal_patch_size=2`, `merge_size=2`, 0.5 mean/std, and ComfyUI passes 16
explicitly rather than inheriting Qwen2-VL's 14. Only the bounds were never
wired to the release's files.

**Read these as pixel counts, not edge lengths.** 65,536 is 256x256;
16,777,216 is 4096x4096. Reading them as edges makes every bound look absurd.

### 3. The floor — the one that is not extreme

The release *enlarges* anything under 65,536 pixels to reach that floor.
ComfyUI's floor is twenty times lower, so it does not.

**Practical impact: any reference under roughly 256x256 equivalent.** A 200x200
thumbnail lands at 192x192, which is 36,864 pixels. The release would enlarge
it; we hand Qwen fewer patches than the vendor intends, so the reference carries
less identity signal. Thumbnails and cropped faces land here easily.

Both towers agree with each other, so this is not a VAE-versus-Qwen split. It is
a ComfyUI-versus-release split, and nothing about it looks extreme. Native
ComfyUI does not warn; this repo's preflight does. **This is the higher-priority
half of the two.**

**Native ComfyUI status: open. Handling in this repo:** preflight reports when
the release floor would enlarge a reference that ComfyUI leaves alone.
`MiniMaxH3ReferenceFit` can deliberately upscale a reference, but it does not
implement the release processor's `min_pixels` rule as a general runtime
boundary and therefore is not a native-equivalence fix.

### 4. The ceiling — real but genuinely extreme

Past 12,845,056 pixels ComfyUI's helper resizes what the VAE already received,
so the two towers see different resolutions of the same picture. Measured
2026-08-22 with the real `process_qwen2vl_images`:

| source | mode | both towers get | Qwen actually sees | divergent |
|---|---|---|---|---|
| 3840x2160 | match | 1344x768 | 1344x768 | no |
| 6144x2048 | max | 6144x2048 | 6144x2048 | no |
| 6656x2048 | max | 6656x2048 | **6432x1984** | **yes** |
| 7168x2048 | max | 7168x2048 | **6688x1888** | **yes** |
| 4096x4096 | max | 2048x2048 | 2048x2048 | no |

**In `ref_image_size="match"` it is unreachable.** Match scales to the
generation's pixel area, around 1M pixels at 1344x768, twelve times below the
ceiling. No source resolution trips it.

**In `max` mode** it needs roughly 3.06:1 or wider at a 2048 short edge — the
crossing is `12,845,056 / 2048² = 3.0625`. The ceiling counts pixels; the ratio
only names it because the sweep pinned the short edge.

When it fires it is a resolution mismatch, not corruption: Qwen gets a bilinear
downscale of the same picture, with slight aspect drift from independent
rounding to 32. The consequence worth holding is that the VAE's latent is finer
than what Qwen extracted identity from, and nothing says so.

[`bench/results/2026-08-21_shipped_reference_bounds.json`](../bench/results/2026-08-21_shipped_reference_bounds.json): **no shipped graph
reaches it.**

**Native ComfyUI status: open. Handling in this repo:**
`MiniMaxH3ReferenceConditioning.image_policy` selects WHOSE still-image ceiling
applies -- `comfy` (default, no opinion), `encoder`, or `release` -- and
pre-applies the selected policy's bounds before the VAE, keeping both towers on
one resolution. That is an opt-in local guard and it is **off by default**; the
native reference node and every graph left on `comfy` remain exposed.

Until 2026-08-24 this was `MiniMaxH3ReferenceFit.keep_towers_matched`, which
clamped to Comfy's `process_qwen2vl_images` default unconditionally. That is
the right ceiling on a native BF16 path and wrong by orders of magnitude under
the AWQ adapter, and the fit node has no `clip` with which to tell the
difference. `MiniMaxH3ReferenceVideoFit` keeps `keep_towers_matched` and keeps
reading Comfy's default, which is correct there: it is a reporter for
native-core paths.

Video references are canvas-sized at around 1M pixels, far below either
ceiling, so the video bounds do not bite in practice. That is derived from the
geometry rather than measured.

---

## 5, 6, 7. The rest of the reference path

Owner: [`h3_references.md`](h3_references.md).

**5. Soundtracks not truncated.** ComfyUI encodes the whole waveform, at 80 rows
per second of excess. sglang cuts every reference soundtrack to the generated
duration via `ffmpeg -t`, and diffusers does the same. It applies to **both**
material chains there -- a video's soundtrack and a standalone audio reference
alike. *Impact:* wasted rows on any soundtrack longer than the render, and
those rows are attended every step. **And more than rows**: a video
reference advances the packed RoPE cursor by
`max(ref_audio_t, sum of video spans)`, so an over-long soundtrack expands
the reference span and pushes the target streams down the timeline. Both
independent reviews reach this (`internal/codex/2026-08-21_h3-conditioning-qwen-independent-review.md`
section 5.3; the cursor formula is in `internal/gemini/minimax_h3_comfyui_end_to_end_trace_and_gap_analysis.md`).

**Native ComfyUI status: open. Handling in this repo:** every shipped graph now
uses the typed compiler, which derives duration from aligned `frame_count` and
slices every video soundtrack and standalone reference internally. The former
explicit `TrimAudioDuration` nodes were removed, so there is no second widget
to drift. Core remains exposed in a hand-built socket graph; preflight warns
there. This closes the gap for this repo's workflows, not in native ComfyUI.

**Where the gap actually bit is narrower than it looked, found the same day.**
`VHS_LoadVideo` already asks ffmpeg for `frame_load_cap / force_rate` seconds
of audio, so a soundtrack was only ever untrimmed because `frame_load_cap` was
0 -- which it was, on every graph here, until that day. The gap is real for
**standalone** `ref_audio_*` on any loader, and for any graph that leaves the
cap at 0. `h3_references.md` carries the per-path table and the measurement.

**6. Reference media never upscaled — a tradeoff ComfyUI exposes and the vendor
does not.** If the source has fewer pixels than the canvas, ComfyUI uses the
source size rounded to 32. All three vendor implementations put it on the canvas
rule with no such clamp.

**This was filed as a defect and that framing was wrong.** Corrected 2026-08-22
after a user pointed out the cost, which the measurements here already had.
Reference rows are attended at **every sampling step**, so a larger reference is
not a one-off cost, it is a tax on the whole render. From the measured table in
[`h3_references.md`](h3_references.md): upscaling a single 1024x1024 reference
image to 2048 costs **6,144 tokens**, half in the reference block and half in
the conditioner's vision blocks, which read the resized image too. A 960x544
reference video at 124 frames is already 18,870 rows; putting it on a 1344x768
canvas roughly doubles the pixel area and the rows with it.

So deliberately passing downscaled references is a **sound default**, and the
clamp is a knob the vendor's pipeline does not give you.

**What is actually unresolved is narrower.** The cost of upscaling is measured
and large. The benefit is **not measured at all** — no controlled comparison
here shows that a canvas-sized reference conditions better than a downscaled
one, and a rendered pair could not show it anyway, because two arms differing in
reference size are two different samples rather than a good and a degraded
version of one. So the honest state is a measured cost against an unmeasured
benefit, which is a strong argument for the current default.

**The clamp is defensible. The silence is not, and that is the actual defect.**
Nothing reports what resolution a reference finally reached. The node accepts a
reference at any size, conditions on whatever it gets, and emits no signal about
what it did, so a user cannot tell a deliberate downscale from an accidental one
and has no way to find out what it cost. That is the same class as gap 4, where
the two towers silently diverge. It applies to the default too, so users who
never chose the tradeoff are also making it. This repo's custom fit nodes report
the reached resolution; the native node still does not.

**And "H3 handles it fine" is not evidence that nothing was lost.** Recorded
because the temptation to treat it as evidence is strong and this file nearly
did. Anyone reporting it has seen their references at one size only, so they
have no baseline to notice a fidelity loss against. It is compatible both with
"the model is robust here" and with "identity is quietly worse than it would
have been", and a rendered comparison cannot separate them: two arms differing
in reference size are two different samples. The claim that would settle it is a
blind session under [`eval_comparison.md`](eval_comparison.md) section 3, and
nobody has run one.

So the state is a **measured** cost against an **unmeasured** benefit, with no
instrument reporting which regime a given render was in. The first argues for
the current default. The second is why the default should say what it did.

One more thing it means, not urgent: output will not reproduce the vendor's for
the same inputs, which matters only if that is your goal.

**7. Mono reference audio raises.** `_encode_ref_audio` does not upmix, so a
mono waveform produces half the rows the packed layout allocated and the
assignment fails. diffusers and DiffSynth-Studio both expand to stereo first.
*Impact:* a hard crash rather than a bad render, which is the better failure.
Gated by [`bench/check_mono_ref_audio.py`](../bench/check_mono_ref_audio.py).

**Native ComfyUI status: open. Handling in this repo:** the typed compiler duplicates
a mono channel to stereo before `_encode_ref_audio`; stereo passes unchanged,
and more than two channels are refused rather than silently selecting a pair.

---

## 8. The VAE encode question

Owner: [`research/sglang_comparison.md`](research/sglang_comparison.md).

**8. Encode precision, tangled with mean-versus-sample.** sglang keeps the video
VAE fp32-resident and decodes in fp16 autocast, and says why: the VAE also
encodes keyframes. Ours runs fp16 throughout.

The decode half was settled in 2026-08-10 — measured 2-3x cost, no established
benefit — but that decision was scoped to decode, and **the vendor does not
decode in fp32 either**. The open half is encode: reference images, reference
videos and keyframes, whose whole job is identity fidelity, computed once per
render rather than per step. `--fp32-vae` cannot express the split.

Half of this is now measured. [`bench/grade_vae_encoder_precision.py`](../bench/grade_vae_encoder_precision.py) grades the
encoder at the call rather than at a rendered clip
([`bench/results/2026-08-21_vae_encoder_precision.json`](../bench/results/2026-08-21_vae_encoder_precision.json)): fp16 is bit-identical
to itself, fp32 moves the latent, bf16 moves it far further. The second and
third are controls; only the middle is a result.

What stays open is whether that delta is *visible*, and a second variable
tangled with it: sglang **samples** the released posterior under a seed pinned
at 42 for keyframes and reference video, where ComfyUI takes the mean
(`comfy/ldm/minimax/vae.py:685`). Any encode instrument has to separate
precision from mean-versus-sample or it measures neither.

### Removed gap: H3 VAE tiling

This snapshot used to describe tiling as an unrecorded runtime fallback. The
installed H3 VAE does not have that branch: `MiniMaxH3VideoVAE` owns a fixed
spatial tile/overlap policy, always uses its H3 temporal chunking, declares
chunked I/O, and its `encode_tiled`/`decode_tiled` routes back through the same
H3 implementation. Generic OOM retry does not silently switch H3 to another
algorithm. Recording the fixed policy may improve provenance; it is not an
untiled-versus-tiled gap.

---

## 10-13. Architectural, and not user-facing

Owner: [`research/sglang_comparison.md`](research/sglang_comparison.md), which
carries the reasoning. Summarised only so the list is complete:

- **No partition concept.** sglang reads `partition` from `model_index.json`
  and refuses a task the checkpoint does not serve. A plain t2v graph here loads
  `ref2va` and renders. This is the frame for the checkpoint-swap arm: a
  `ref2va` loss at plain t2v is *not a t2v model by its own release metadata*,
  not a defect.
- **Exact AdaLN cache.** Buys unpruned accuracy at pruned memory, not speed.
  Downstream of `open_experiments.md` #22.
- **Breakable CUDA graphs.** No equivalent here. Their own quality gate requires
  it off.
- **Step caching.** Priced and declined 2026-08-20: their speedup came from
  skipping steps on a 16-step schedule at 50 steps on four H200s. At 4 steps
  there is nothing to skip.

**The filter that applies to all four.** sglang's audited `quality="high"`
deployment is a 4xH200 server with sequence parallelism we cannot copy on one
card, and its own gate requires quantization, `torch.compile` and layerwise
offload all *off*. "sglang has it and we don't" is not on its own an argument.

---

## Settled — recorded so nobody re-derives them

- **The 2 fps conditioner subsample**, including the index-list pad to the
  temporal patch and the merged-pair timestamp: **confirmed identical**.
- **Patch geometry**: **confirmed identical**, and ComfyUI correctly passes 16
  rather than inheriting Qwen2-VL's 14.
- **qkv row-permutation as the cause of the fp8/int8 fidelity gap**:
  **refuted** on two independent grounds. The fp8 file has a scalar
  `weight_scale` with no row to misalign, and the gap is uniform across module
  kinds rather than confined to the fused qkv weight.
- **What sglang does with a reference video shorter than its aligned target**:
  **read, not verified.** It truncates with `ffmpeg -frames:v`; whether the
  encoder then re-aligns was not traced. Cheap to close.

---

## What we do that they do not

Recorded so the comparison is not read one-directionally. Sol-Attn, the sage
kernels and the SLA router have no counterpart in sglang's H3 path, which runs
dense FlashAttention. [`SOLATTN.md`](SOLATTN.md) owns those numbers. Their
speedups and ours are not comparable and must not be put in the same table.

---

## Everything built against these gaps

Code, instruments and records, per gap. Except where a row explicitly says
"native ComfyUI", entries in these tables are this repository's handling and
do not change the native status above. A gap with nothing in this table is a gap
nothing is watching.

### 1. Special tokens

| what | where |
|---|---|
| The merged native fix in this installed ComfyUI | [PR 15808](https://github.com/Comfy-Org/ComfyUI/pull/15808), `comfy/text_encoders/minimax.py` at installed commit `924743af` |
| Tokenization audit, nine scenes plus reference integrity | [`bench/audit_h3_marker_tokenization.py`](../bench/audit_h3_marker_tokenization.py) |
| Record | [`bench/results/2026-08-24_h3_marker_tokenization_native.json`](../bench/results/2026-08-24_h3_marker_tokenization_native.json) |
| Encoder-level companion, what the states do | [`bench/grade_h3_marker_tokens.py`](../bench/grade_h3_marker_tokens.py), [record](../bench/results/2026-08-21_h3_marker_token_states.json) |
| Are the embedding rows trained | [`bench/audit_h3_token_embeddings.py`](../bench/audit_h3_token_embeddings.py) |
| Saved-graph compatibility tombstone; no tokenizer mutation | [`vendor_tokens.py`](../vendor_tokens.py) |
| The release's declared list, never retyped | [`vendor_config.py`](../vendor_config.py), guarded by [`bench/check_vendor_config.py`](../bench/check_vendor_config.py) |

The audit's two controls are what make it worth running. A marker-free prompt
must tokenize identically across the release, native and reconstructed legacy
tokenizers, and the native arm must reproduce the **release** tokenizer exactly
rather than merely differing from legacy. Missing native tokens refuse the run.

### 2. Frame rate

| what | where |
|---|---|
| Fails the build if a loader feeding a reference socket drops off 24 | [`bench/check_ref_prompt_labels.py`](../bench/check_ref_prompt_labels.py) |
| Prompt and label grading before a render is queued | [`bench/preflight_graph.py`](../bench/preflight_graph.py) |
| Typed ownership and normalization | [`reference_conditioning.py`](../reference_conditioning.py), controlled by [`bench/check_reference_runtime.py`](../bench/check_reference_runtime.py) |

Native ComfyUI has no fix. The legacy check enforces this repo's workaround;
the typed surface implements the behaviour locally.

### 3 and 4. Pixel bounds

| what | where |
|---|---|
| Where the ceiling actually resizes, against the real helper | [`bench/measure_qwen_bounds_bite.py`](../bench/measure_qwen_bounds_bite.py), [record](../bench/results/2026-08-21_qwen_bounds_bite.json) |
| Does any shipped graph hand Qwen a reference it shrinks | [`bench/audit_shipped_reference_bounds.py`](../bench/audit_shipped_reference_bounds.py), [record](../bench/results/2026-08-21_shipped_reference_bounds.json) |
| The release's bounds, read not retyped | [`vendor_config.py`](../vendor_config.py) `image_pixel_bounds()` / `video_pixel_bounds()` |

The measurement instruments **report**; neither refuses. Preflight reports a
real graph crossing either divergent bound, but there is no runtime assertion
that implements the release's image floor.

### 7. Mono reference audio

| what | where |
|---|---|
| Gate | [`bench/check_mono_ref_audio.py`](../bench/check_mono_ref_audio.py) |
| Reports it on a real graph's real media | [`bench/preflight_graph.py`](../bench/preflight_graph.py) |
| Typed mono-to-stereo boundary | [`reference_conditioning.py`](../reference_conditioning.py), controlled by [`bench/check_reference_runtime.py`](../bench/check_reference_runtime.py) |

Its two red states mean opposite things and the file says which: a failing mono
arm is the defect, a *succeeding* one means upstream fixed it and the check
should be retired rather than repaired.

**Still open in native ComfyUI; locally handled without graph-wide channel
nodes on the typed surface.** Preflight warns on legacy mono media. For typed
graphs it reports the compiler's upmix instead. "Channel count unreadable"
remains its own state -- no ffprobe and no audio stream are not a pass.

### 8. VAE encode precision

| what | where |
|---|---|
| Grades the encoder at the call, not at a rendered clip | [`bench/grade_vae_encoder_precision.py`](../bench/grade_vae_encoder_precision.py) |
| Record | [`bench/results/2026-08-21_vae_encoder_precision.json`](../bench/results/2026-08-21_vae_encoder_precision.json) |
| The two probe arms | `workflows/h3_probe_ref_vae_encoder_fp16_api.json`, `workflows/h3_probe_ref_vae_encoder_fp32_api.json` |

The probe arms render and **assert nothing**; they price the knob and prove it
runs. They are labelled as unable to say which output is better, because a
rendered clip cannot A/B a numerical change.

### 6. Reference video sizing and reporting

| what | where |
|---|---|
| The node that was missing entirely | [`reference_video_fit.py`](../reference_video_fit.py) |
| Holds its copy of core's sizing rule to core's real behaviour | [`bench/check_ref_video_prediction.py`](../bench/check_ref_video_prediction.py) |
| Opt-in local release policy | `MiniMaxH3ReferenceConditioning.video_policy=release`, controlled by [`bench/check_reference_runtime.py`](../bench/check_reference_runtime.py) and its red harness |
| Shipped hybrid encoder policy | `video_policy=encoder`: native-compatible no-upscale VAE geometry plus the selected encoder artifact's snapshotted duration-aware Qwen stage, implemented locally and controlled against accidental release-config substitution by the same runtime check |

### Custom W4A16 encoder format (adjacent, not a vendor-release gap)

| what | where |
|---|---|
| Native NVFP4-AWQ control and local compressed-tensors W4A16 loader contract | [`bench/check_h3_awq_encoder.py`](../bench/check_h3_awq_encoder.py) |
| Repo-local loader/adaptation | [`h3_awq_encoder.py`](../h3_awq_encoder.py) (`MiniMaxH3AWQEncoderLoader`) |
| Detailed native/local boundary and checkpoint comparison | [`docs/h3_awq_encoder.md`](h3_awq_encoder.md) |
| Exact source artifact configs, recipes and digests | [`config/qwen3vl_32b_minimax_h3_w4a16_awq/`](../config/qwen3vl_32b_minimax_h3_w4a16_awq/) |

### The contracts underneath all of it

| what | where |
|---|---|
| Core's reference-node contracts | [`bench/check_reference_contracts.py`](../bench/check_reference_contracts.py) |
| Ordered plan against core plus intended differences | [`bench/check_reference_order.py`](../bench/check_reference_order.py) |
| Typed runtime media and compiler contracts | [`bench/check_reference_runtime.py`](../bench/check_reference_runtime.py), red harness [`bench/red/show_red_reference_runtime.py`](../bench/red/show_red_reference_runtime.py) |
| Label/preflight visibility of typed chains | [`bench/check_typed_reference_consumers.py`](../bench/check_typed_reference_consumers.py) |

All seven contracts in `MiniMaxH3ReferenceToVideo` are controlled. The ordered
surface preserves the four that remain semantic contracts and deliberately
replaces suffix pairing and fixed modality grouping with ownership and list
position.

### Policy disposition

Soundtrack duration and mono normalization are executable properties of this
repo's typed runtime; native ComfyUI remains unchanged. VAE tiling was removed
as a gap after the native H3-owned paths were traced.

The release's video upscale and duration-aware Qwen resize are implemented as
one **opt-in local policy**. `video_policy=release` puts the full-rate VAE view
on the release canvas and independently runs the raw 2 fps samples through the
release's own Qwen video processor. Shipped graphs now default to the local
`video_policy=encoder` hybrid: it keeps ComfyUI's cheaper no-upscale VAE view
while applying the custom encoder's source-config, duration-aware Qwen stage.
`video_policy=comfy` remains available as the unmodified native preprocessing
control. Native
`MiniMaxH3ReferenceToVideo` still does neither stage and exposes no policy;
this implementation therefore does not close gap 6 upstream.

Gap **6** needs no further instrument for the cost or mechanism, which are now
reported and live-smoked; what remains is a controlled comparison of the
*benefit*, and
[`eval_comparison.md`](eval_comparison.md) section 3 is the only process here
that could supply one. Nobody has asked for it, and the measured cost argues
against bothering.


---

## Native status versus what this repo enforces

A gap with no assertion behind it is a gap that will come back.

| gap | native ComfyUI status | this repo's enforcement or handling |
|---|---|---|
| 1, special tokens | **fixed** in installed commit `924743af` / merged PR 15808 | [`bench/audit_h3_marker_tokenization.py`](../bench/audit_h3_marker_tokenization.py) requires and verifies native behavior |
| 2, frame rate | open | legacy: [`bench/check_ref_prompt_labels.py`](../bench/check_ref_prompt_labels.py); typed: [`bench/check_reference_runtime.py`](../bench/check_reference_runtime.py) |
| 3, image floor | open | preflight warning only; **no runtime parity enforcement** |
| 4, image ceiling | open | opt-in local guard, **off by default**: `MiniMaxH3ReferenceConditioning.image_policy` (`encoder`/`release`) since 2026-08-24, replacing `MiniMaxH3ReferenceFit.keep_towers_matched`, which read the wrong ceiling under the AWQ adapter. Graphs left on `comfy` remain exposed |
| 5, soundtrack length | open | shipped typed graphs: [`bench/check_reference_runtime.py`](../bench/check_reference_runtime.py); native socket graphs: [`bench/preflight_graph.py`](../bench/preflight_graph.py) reports required upstream handling |
| 6, media upscale/reporting | sizing divergence remains; native path is silent | custom fit nodes report the resolution reached; the typed conditioner's opt-in `release` policy handles both video stages locally, with no claim that native sizing now matches the vendor |
| 7, mono audio | open | native defect gate: [`bench/check_mono_ref_audio.py`](../bench/check_mono_ref_audio.py); local typed handling: [`bench/check_reference_runtime.py`](../bench/check_reference_runtime.py) |
| 8, VAE encode precision | open | **nothing enforces a choice**; measurement only |
| 9, VAE tiling | not a gap in installed native H3 path | policy documented from native source; no custom fix |

The image floor and the visual VAE mean-versus-sample question remain the
highest-value unclosed conformance items. Neither blocks typed migration.
