# Where ComfyUI's H3 path differs from the vendor's

last updated: 2026-08-22

Every known divergence between this ComfyUI install and the MiniMax H3 release,
in one place, with what each one costs a working user.

**This file is a snapshot, not an authority.** Every fact in it is owned by
another document, named in each section. It exists because the ownership rule
that keeps those documents from drifting also meant that answering "what are all
the gaps" required knowing which of three files to open. **If this file
disagrees with an owner, the owner is right** and this one is stale. Regenerate
it by re-reading the owners rather than by editing it in place.

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

## Summary

Priority is by what it costs a working user, not by how interesting it is.

| # | gap | kind | status |
|---|---|---|---|
| 1 | Seven special tokens absent from the tokenizer | config | **fixed upstream, PR open** |
| 2 | Reference video frame rate assumed, not enforced | behavioural | open, workaround gated |
| 3 | Reference image floor (`min_pixels`) | config | open, unenforced |
| 4 | Reference image ceiling (`max_pixels`) | config | **guarded in this pack**, open in core |
| 5 | Reference soundtracks not truncated | behavioural | open by choice |
| 6 | Reference media never upscaled, and never reported | behavioural | clamp is a knob; **the silence is the defect** |
| 7 | Mono reference audio raises | behavioural | gated |
| 8 | VAE encode precision, and mean vs sample | behavioural | half measured |
| 9 | VAE tiling unrecorded | behavioural | open, unenforced |
| 10-13 | Partition gate, AdaLN cache, CUDA graphs, step caching | behavioural | architectural, not user-facing |

---

## 1. Seven special tokens — fixed upstream

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
([`bench/results/2026-08-22_h3_marker_tokenization.json`](../bench/results/2026-08-22_h3_marker_tokenization.json)): in an ordinary
two-person dialogue prompt, 9 of 91 non-marker tokens come out different.

**Status: fixed by [Comfy-Org/ComfyUI PR 15808](https://github.com/Comfy-Org/ComfyUI/pull/15808)**
(kijai), which declares the seven on a `Qwen3VLSDTokenizer` subclass so every
consumer gets them, including core's `MiniMaxH3ReferenceToVideo`. **Open, not
merged**, so an install without it still has the defect.

Verified here against the PR's own diff applied to a clean master: all nine
scenes reproduce the release tokenizer's ids exactly, the reference path carries
the marker beside its vision blocks, and a marker-free prompt is byte-identical
before and after. [`bench/audit_h3_marker_tokenization.py`](../bench/audit_h3_marker_tokenization.py) is the harness and
runs identically with or without the patch.

`MiniMaxH3VendorTokens` in this pack is deprecated by that fix.
`clip_with_vendor_tokens` is not, because a pack cannot assume the install it
runs on carries the patch.

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
mechanism is `qwen35.py:660-663`: `cu_seqlens` splits attention at every
`h*w`, so attention never crosses a frame in either presentation, and the
position construction carries no temporal term.

**Random weights are the right instrument and not a shortcut**: the claim is
about how the tower routes tokens, which is architecture rather than anything
learned. It tests nothing weight-dependent, and nothing in the claim is.

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
a ComfyUI-versus-release split, and nothing about it looks extreme, and nothing
warns. **This is the higher-priority half of the two.**

`MiniMaxH3ReferenceFit` does not close it, because it runs before the tokenizer.

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

**Closed in the graphs on 2026-08-22, not in the node.** `TrimAudioDuration` at
`length / 24` sits on every `ref_audio_*` and `ref_video_audio_*` socket here.
Core is unchanged, so a hand-built graph is still exposed; the control is
`bench/preflight_graph.py`, which warns on an untrimmed socket and on a baked
trim that disagrees with `length`.

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
the two towers silently diverge, and gap 9, where a tiled decode is
indistinguishable afterwards. It applies to the default too, so users who never
chose the tradeoff are also making it.

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

---

## 8 and 9. The VAE

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

**9. VAE tiling is silent.** ComfyUI tiles under memory pressure and records
nothing; sglang treats decode mode as part of the quality contract and refuses
modes it considers inexact. *Impact:* a tiled decode cannot be distinguished
from an untiled one after the fact, so a render's provenance is incomplete.

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

Code, instruments and records, per gap. A gap with nothing in this table is a
gap nothing is watching.

### 1. Special tokens

| what | where |
|---|---|
| The upstream fix | [PR 15808](https://github.com/Comfy-Org/ComfyUI/pull/15808), `comfy/text_encoders/minimax.py` |
| Tokenization audit, nine scenes plus reference integrity | [`bench/audit_h3_marker_tokenization.py`](../bench/audit_h3_marker_tokenization.py) |
| Record | [`bench/results/2026-08-22_h3_marker_tokenization.json`](../bench/results/2026-08-22_h3_marker_tokenization.json) |
| Encoder-level companion, what the states do | [`bench/grade_h3_marker_tokens.py`](../bench/grade_h3_marker_tokens.py), [record](../bench/results/2026-08-21_h3_marker_token_states.json) |
| Are the embedding rows trained | [`bench/audit_h3_token_embeddings.py`](../bench/audit_h3_token_embeddings.py) |
| The pack's shim, still needed on an unpatched install | [`vendor_tokens.py`](../vendor_tokens.py) (`clip_with_vendor_tokens`), consumed at [`conditioning.py`](../conditioning.py) |
| The release's declared list, never retyped | [`vendor_config.py`](../vendor_config.py), guarded by [`bench/check_vendor_config.py`](../bench/check_vendor_config.py) |

The audit's two controls are what make it worth running. A marker-free prompt
must tokenize identically across all three tokenizers, and the corrected arm
must reproduce the **release** tokenizer exactly rather than merely differing
from stock. It also asserts that `clip_with_vendor_tokens` is a **no-op** when
the core patch is present, and refuses the run if the shim quietly does work
instead.

### 2. Frame rate

| what | where |
|---|---|
| Fails the build if a loader feeding a reference socket drops off 24 | [`bench/check_ref_prompt_labels.py`](../bench/check_ref_prompt_labels.py) |
| Prompt and label grading before a render is queued | [`bench/preflight_graph.py`](../bench/preflight_graph.py) |

No fix. The check enforces the workaround, not the behaviour.

### 3 and 4. Pixel bounds

| what | where |
|---|---|
| Where the ceiling actually resizes, against the real helper | [`bench/measure_qwen_bounds_bite.py`](../bench/measure_qwen_bounds_bite.py), [record](../bench/results/2026-08-21_qwen_bounds_bite.json) |
| Does any shipped graph hand Qwen a reference it shrinks | [`bench/audit_shipped_reference_bounds.py`](../bench/audit_shipped_reference_bounds.py), [record](../bench/results/2026-08-21_shipped_reference_bounds.json) |
| The release's bounds, read not retyped | [`vendor_config.py`](../vendor_config.py) `image_pixel_bounds()` / `video_pixel_bounds()` |

Both instruments **report**; neither refuses. The floor has no instrument at
all — everything above measures the ceiling.

### 7. Mono reference audio

| what | where |
|---|---|
| Gate | [`bench/check_mono_ref_audio.py`](../bench/check_mono_ref_audio.py) |
| Reports it on a real graph's real media | [`bench/preflight_graph.py`](../bench/preflight_graph.py) |

Its two red states mean opposite things and the file says which: a failing mono
arm is the defect, a *succeeding* one means upstream fixed it and the check
should be retired rather than repaired.

**Still open on purpose, and preflight only reports it.** Since 2026-08-22
preflight ffprobes the media each ref-audio socket actually resolves to and
warns when it is mono. It does not upmix: that belongs in core's
`_encode_ref_audio`, and a `JoinAudioChannels(a, a)` in every shipped graph
would alter every stereo source to prevent a crash none of them hit. "Channel
count unreadable" is reported as its own state -- no ffprobe and no audio
stream are not a pass.

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

### The contracts underneath all of it

| what | where |
|---|---|
| Core's reference-node contracts, asserted for the first time | [`bench/check_reference_contracts.py`](../bench/check_reference_contracts.py) |

Five of the seven contracts in `MiniMaxH3ReferenceToVideo` gained a control on
2026-08-22, after standing enforced by nothing through two postmortems. Two
remain uncovered and the check prints which on every run.

### Nothing built

Gaps **5** (soundtrack length) and **9** (VAE tiling) have no instrument, no
check and no fix. They are recorded here and nowhere else in executable form.

Gap **6** needs no instrument for the cost, which is already measured; what it
would need is a controlled comparison of the *benefit*, and
[`eval_comparison.md`](eval_comparison.md) section 3 is the only process here
that could supply one. Nobody has asked for it, and the measured cost argues
against bothering.


---

## What is actually enforced

A gap with no assertion behind it is a gap that will come back.

| gap | enforced by |
|---|---|
| 1, special tokens | [`bench/audit_h3_marker_tokenization.py`](../bench/audit_h3_marker_tokenization.py), plus the upstream fix |
| 2, frame rate | [`bench/check_ref_prompt_labels.py`](../bench/check_ref_prompt_labels.py) |
| 3, image floor | **nothing** |
| 4, image ceiling | **guarded in this pack since 2026-08-22**: `MiniMaxH3ReferenceFit`'s `keep_towers_matched` holds the reference under Qwen's ceiling so both towers agree. Core is still unguarded for anyone not wiring that node |
| 5, soundtrack length | **nothing** |
| 6, media upscale | the clamp needs none. The **silence** is now addressed: both reference nodes report the resolution reached, and `MiniMaxH3ReferenceVideoFit` covers video, which nothing touched before |
| 7, mono audio | [`bench/check_mono_ref_audio.py`](../bench/check_mono_ref_audio.py) |
| 8, VAE encode precision | **nothing** |
| 9, VAE tiling | **nothing** |

Five of nine are enforced by nothing, and the two highest-priority open gaps —
the image floor and the VAE encode question — are both among them.
