# MiniMax H3 references: every type, what it costs, how to prompt it

`ref2va` conditions on an ordered list of references. This is what each type
is, what ComfyUI actually does to it, what it costs, and how to write the
prompt so the model uses it the way you meant.

last updated: 2026-08-25

Sources: MiniMax's official prompt guide, general prompting research, ComfyUI's
own code, and **sglang's MiniMax H3 serving path** (`coderef/sglang`, read at
commit `a41da991c8`), which is the vendor-side authority this document compares
against. Every number marked **measured** was taken on this install against a
live render; everything else is read from source and says so.

**"The reference pipeline" meant diffusers here until 2026-08-21, and that was
the wrong authority.** We do not run diffusers. Its H3 modular pipeline is a
portability target — the thing `h3_rules.reference_would_emit()` answers about,
and the reason 345 rather than 362 keeps appearing in this repo — not a
description of how the release is served. Every vendor-side comparison below
that has been re-derived against sglang says **sglang** and carries a citation.
Anything still saying *the reference pipeline* is a claim about diffusers alone,
not yet checked against sglang, and should be read that way until it is.

Written 2026-08-13 against ComfyUI v0.33.0. Reference-image sizing re-read from
source and corrected 2026-08-16; the vendor-side image path re-derived against
sglang 2026-08-21.

---

## The four reference types: native sockets and this repo's typed surface

Native ComfyUI's `MiniMaxH3ReferenceToVideo` has four parallel reference socket
groups. There is **no mask socket and no fps input** on any of them.

| socket | type | max | what it is |
|---|---|---|---|
| `ref_images.ref_image_N` | IMAGE | 9 | a still, at high detail |
| `ref_videos.ref_video_N` | IMAGE (frame batch) | 3 | a clip, on the canvas rule |
| `ref_video_audios.ref_video_audio_N` | AUDIO | 3 | the soundtrack of the **same-numbered** video |
| `ref_audios.ref_audio_N` | AUDIO | 3 | a standalone audio asset |

Shipped workflows in this repo no longer wire those groups. They use local
`MiniMaxH3AppendRefImage`, `MiniMaxH3AppendRefVideo`, and
`MiniMaxH3AppendRefAudio` nodes feeding `MiniMaxH3ReferenceConditioning`.
List position is explicit, the video record owns its optional soundtrack and
`VHS_VIDEOINFO`, and the compiler owns duration/channel normalization. This is
the repo's handling of native gaps, not a claim that core's socket node changed.

The generator deliberately preserves native presentation order -- images,
then each sounded video, then standalone audio -- so existing prompts keep
their ordinals. A hand-built typed chain can choose another order.

Two limits **diffusers** enforces with a raise and ComfyUI does not: 12
references total across all types (`coderef/diffusers/src/diffusers/modular_pipelines/minimax_h3/before_encoder.py:410-413`), and an audio reference
paired with nothing (`coderef/diffusers/src/diffusers/modular_pipelines/minimax_h3/before_encoder.py:414-418`). Wire 15 and ComfyUI will accept it.

**Do not read those as vendor-wide — re-derived 2026-08-21 and only one of
three vendor implementations has them.** sglang's ref2va profile sets neither a
minimum nor a maximum condition count (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py:183-231`), and
DiffSynth-Studio's H3 pipeline raises only on an unknown reference kind and on
a silent video passed as `video` (`coderef/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py`). Both are exactly as permissive as
ComfyUI here. diffusers is transcribing a documented API limit into a local
guard — its own docstring says the limits "bound nothing but this block's own
validation" — and the origin is MiniMax's README table for the Open Platform
API, `coderef/MiniMax-H3/README.md:85`. That table also carries per-clip
2-15 s and aggregate 15 s duration limits for video and audio references, which
nothing in ComfyUI enforces either.

**A fourth implementation states them per type, and calls them the
checkpoint's.** `aigc-apps/VideoX-Fun` PR #508 (merged; not in `coderef/`, read
from the diff 2026-08-27) adds `examples/minimax_h3/predict_ref2va.py`, whose
comment reads: "Budgets of the released checkpoint: at most 9 images, 3 videos,
3 audios and 12 references in total, and an audio reference cannot stand alone."
Same 12 and same audio rule diffusers raises on, plus a per-type split nothing
else here records -- from the org that published the PDD LoRAs.

Weigh it as a claim, not a control: it is a comment in an example script, so
that repo documents the budgets rather than enforcing them, and it does not say
where the per-type numbers come from. It does move the open question above --
whether these are vendor-wide or one validator's transcription -- because the
publisher's own example now asserts them of the checkpoint. Nothing in ComfyUI
enforces any of it either way.

A reference video is an **IMAGE batch, not a VIDEO** — it arrives through a
frame loader, which is why the frame rate is your problem (below).

---

## What ComfyUI does to each one

### Image references

Scaled and rounded to 32. **ComfyUI clamps the scale with `min(1.0, ...)` in
both of its modes and sglang does not** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:125-177`,
read from source 2026-08-21: `scale = 2048 / min(w, h)`, `"allow_upscale": True`
in the returned shape, nearest-32 per axis, and a docstring that says in as many
words that reference images have no area-cap branch). So a reference smaller
than 2048 on its short side reaches the DiT under-sized — and identity fidelity
is the whole job of a reference image. Native ComfyUI still behaves this way;
this repo's optional `MiniMaxH3ReferenceFit` node handles the divergence only
for graphs that explicitly wire it.

**The default mode is off-vendor in the other direction too, and by more.**
`ref_image_size` defaults to `match`, which sizes a reference to the
*generation's pixel area*. sglang's ceiling does not move with the canvas: it is
a fixed 2048 short edge with no area cap, so a 16:9 reference lands near 7.5 MP
where `match` on a 1344x768 render lands near 1 MP. `max` fixes the ceiling and
`MiniMaxH3ReferenceFit(allow_upscale=True)` fixes the floor; you need both to
condition the way the release is served.

Two separate knobs decide the final size and they are constantly confused for
each other. See **Sizing a reference image** below; the short version is that
`ref_image_size` never upscales in either of its modes, so `max` alone does
nothing for a reference that is already under 2048.

Refused outside 1:4..4:1. Image references are deliberately **exempt from the
768x1344 area cap** that binds the video, which is why one can legitimately
reach 7.5 megapixels when the video cannot exceed about one.

### Video references

1. Canvas from the reference's **own** aspect ratio via `adapt_canvas`.
2. **Never upscaled.** If the source has fewer pixels than that canvas,
   ComfyUI uses the source size rounded to 32 (`comfy_extras/nodes_minimax_h3.py:316-320`).
   **All three vendor implementations put it on the canvas rule with no such
   clamp** (re-derived 2026-08-21): sglang resolves reference-video geometry
   through the same shape function it uses for the target, from the clip's own
   display aspect (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/prequeue.py:292-302`); DiffSynth-Studio scales by
   `min(768/min(w,h), sqrt(max_pixels/(w*h)))` with no unity clamp; diffusers
   routes through its canvas resolver. Native ComfyUI still has this
   divergence. This repo's typed conditioner can handle it locally with
   `video_policy=release`, which also enables the coupled Qwen stage below;
   generated graphs use `video_policy=encoder`, which keeps native-compatible
   VAE sizing while enabling only the source-config Qwen stage.
3. Truncated to the **generated** frame count, then snapped **down** to the
   `17n+5` grid. Fewer than 5 frames raises.
4. VAE-encoded whole. Those rows ride **every sampling step**.
5. Subsampled to **2 fps** for the text conditioner, each merged frame pair
   labelled `<T.T seconds>`.

Consequence worth internalising: **a reference video is truncated to the
generated frame count**, so a short render can only ever be conditioned on a
short reference. Rendering 124 frames means at most 124 frames of reference no
matter how long the clip is.

### Audio references

Resampled to the audio VAE's rate. **Core does not truncate.** ComfyUI encodes
the whole waveform, at 80 rows per second of excess.

**Rows are not the only consequence, and framing it as cost alone was too
narrow.** A reference block advances the packed 3D-RoPE cursor, and a video
reference advances it by `max(ref_audio_t, sum of video spans)` -- so a
soundtrack longer than its own visual reference **expands the reference span**
and pushes the target streams further down the timeline. Both independent
reviews of this pipeline reach it: `internal/codex/2026-08-21_h3-conditioning-qwen-independent-review.md`
section 5.3 states it directly, and `internal/gemini/minimax_h3_comfyui_end_to_end_trace_and_gap_analysis.md`
carries the cursor formula. Trimming is therefore a correctness change on the
soundtrack path, not only a saving.

**`length / 24` is not an approximation of the right value, it is the exact
one**, and the arithmetic says so. `_ref_t_span` compares `ref_audio_t`
against `sum(_video_t_spans(latent_t))` in the same timeline units
(`comfy/ldm/minimax/model.py:105-113`). At 124 frames those are 207 and
206.667; at 362 they are 603 and 603.333. A soundtrack trimmed to the render
lands on its own video's span and the `max` is a tie. Left untrimmed, the
19.541s source clip here gives `ref_audio_t` about 782 against a video span of
206.667 -- the reference block claims **3.8x** the timeline it should, and the
target streams start that much later. Computed 2026-08-22 with
`temporal_shape` and `_video_t_spans` imported, not restated.

**Every shipped graph here is now capped at the typed boundary.** The compiler
derives `frame_count / 24`, slices both an owned video soundtrack and standalone
audio, duplicates mono to stereo, and refuses ambiguous multichannel input.
There is no `TrimAudioDuration` widget left to drift when a bench patches
`length`. **Core itself still does none of this**, so a native socket graph
remains exposed; preflight reports its trim and channel state separately.

**On the video-soundtrack path the trim overlaps a mechanism VHS already had,
and this was found after wiring it.** `VHS_LoadVideo` asks ffmpeg for
`frame_load_cap * (1 / force_rate)` seconds of audio
(`custom_nodes/comfyui-videohelpersuite/videohelpersuite/load_video_nodes.py:402`
into `custom_nodes/comfyui-videohelpersuite/videohelpersuite/utils.py:224-233`,
paths relative to the ComfyUI root), so with
`frame_load_cap` set to the generated length the soundtrack arrives already
capped before the typed compiler sees it. Measured 2026-08-22 by
driving that exact ffmpeg call: cap 0 yields 19.541s of a 19.541s track, cap
124 yields 5.167s, cap 362 yields 15.083s.

So on that path VHS and the typed compiler handle the still-open native gap
**redundantly**, and it is worth keeping which is doing what straight:

| path | locally handled by `frame_load_cap` | locally handled by typed compiler |
|---|---|---|
| `ref_video_audio_*` | yes, whenever the cap is non-zero | yes, and it is what holds if the cap goes back to 0 |
| `ref_audio_*` (standalone `LoadAudio`) | no such mechanism exists | **yes, and it is the only one** |

The compiler cap stays authoritative on both. It survives a loader cap someone
sets back to zero and is the only mechanism on the standalone path.

**Two of three vendor implementations truncate; one does not** (re-derived
2026-08-21). sglang cuts every reference soundtrack to the generated duration —
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/audio_encoding.py:147-151` computes it from the frame count and
it reaches `ffmpeg -t` — and diffusers does the same at `coderef/diffusers/src/diffusers/modular_pipelines/minimax_h3/before_encoder.py:346`.
DiffSynth-Studio does not truncate, matching ComfyUI. Worth knowing that
ComfyUI is inconsistent with *itself* here: a **keyframe's** audio is truncated
to the remaining track and raises past the end
(`comfy_extras/nodes_minimax_h3.py:226-230`), and the reference path in the
same file does neither.

---

## Frame rate: the one that bites silently

ComfyUI's node has **no fps input** and assumes 24 twice over — for the DiT's
temporal clock and for the `<T.T seconds>` labels the conditioner reads
(`FPS = 24` at `comfy_extras/nodes_minimax_h3.py:30`; no node in that file
exposes an fps or rate input).

**Both implementations target 24. Only one enforces it** (read 2026-08-22 at
`coderef/sglang` `a7ec6b97f7`). sglang's constant is the same value —
`MINIMAX_H3_SUPPORTED_FPS = 24` at
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py:29`
— but it reaches the clip as an ffmpeg `fps=` filter in the same decode pass
that does rotation, Lanczos scaling and square-pixel normalisation
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:397`),
so a 30 fps source is genuinely converted to constant-rate 24 before anything
else sees it. ComfyUI applies no such filter: it takes whatever the loader
hands it and assumes the rate.

Two consequences follow from that being one ffmpeg pass. **fps is never in
sglang's API surface** — the caller asks for `duration_seconds` and the frame
count is derived as `duration * 24` at request validation
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py:291-295`),
so there is no rate for a user to get wrong. And **the decoded array is shared
by Qwen and the visual VAE** by construction rather than by convention
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:380-383`), which is the same one-tensor-two-towers
property ComfyUI gets by passing one object to both.

**Measured**, three 6.00-second clips differing only in frame rate:

| source | `force_rate` | H3 reads it as | error | last label |
|---|---|---|---|---|
| 24 fps | 0 or 24 | 5.875s | 0.0% | `<5.2 seconds>` |
| 25 fps | 0 | 5.875s | **+4.2%** | `<5.2 seconds>` |
| 30 fps | 0 | **7.292s** | **+25.0%** | `<7.0 seconds>` |
| any | 24 | correct | 0.0% | correct |

At 30 fps the model is told a six-second reference is seven and a quarter
seconds of action. **Set `force_rate=24` on the loader.** A 24 fps source is
unaffected either way, which is exactly why testing on one proves nothing.
`bench/check_ref_prompt_labels.py` fails the build if any loader feeding a
reference socket drops off 24.

---

## What references cost

Reference rows are attended at **every sampling step**, exactly as video rows
are. Measured against a live render, 1344x768:

| reference | rows |
|---|---|
| audio, per second | 80 |
| image at `match` | ~1,008 |
| image, 1024x1024 source, **upscaled** to 2048 | 4,096 |
| image, 1280x720 source, **upscaled** to 2048 | 7,296 |
| video, 960x544 source, 124 frames | 18,870 |
| video, 960x544 source, 345 frames | **52,020** |

> **Corrected 2026-08-16.** The two image rows above used to read "at `max`",
> which credited our node's effect to core. `max` alone cannot produce either
> number: it clamps with `min(1.0, ...)`, so a 1024x1024 source at `max` is
> **1,024 rows, unchanged**. 4,096 and 7,296 are what you get with
> `MiniMaxH3ReferenceFit(allow_upscale=True)` feeding `max`. The next section
> is why.

**Reference IMAGES cost in two places as well, and the table above is only
the first.** Measured 2026-08-13 on one graph, toggling `allow_upscale` alone,
from two Preflight lines:

| segment | refs not upscaled | upscaled to 2048 | delta |
|---|---|---|---|
| references | 53,044 | 56,116 | **+3,072** |
| text | 9,294 | 12,366 | **+3,072** |
| total | 143,386 | 149,530 | **+6,144** |

The reference-block half is exactly what the table predicts — a 1024x1024
source is 1,024 rows and a 2048x2048 one is 4,096. The text segment grew by
the same amount again, because the conditioner's vision blocks read the
resized image too. **So upscaling one reference image cost 6,144 tokens, not
3,072**, and any budget built from the table alone is short by half.

**And the text half scales with reference COUNT, measured across a ladder.**
Reported 2026-08-15 by the single-frame session, from Preflight lines on
1/2/3/4/6 references at the shipped sizing. Their measurement, not re-derived
here:

| references | text rows | reference rows |
|---|---|---|
| 1 | 4,171 | 4,096 |
| 4 | 19,778 | 19,648 |
| 6 | 28,002 | 27,840 |

Two things follow. The text segment does not merely grow, it lands **75-160
rows above the reference segment itself** at every rung, so "images cost
double" is if anything an understatement. And the prompt is under 200 tokens of
that text column, so the column is almost entirely vision blocks.

**Nine references at the shipped sizing is roughly 94,000 rows and OOMs on a
24 GB card**, which is larger than the whole 124-frame video graph. Same
source, same date.

**A reference video costs rows in two places, not one.** The DiT reference
block is the number above; the conditioner also reads the clip at 2 fps and
each merged frame pair becomes a vision block **inside the text segment**, at
roughly 519 tokens per block. Going from 124 to 345 reference frames grew the
text segment by 4,667 tokens on top of the 33,150 extra reference rows.

### It does not all fit

**Measured on a 24 GB card**, images at `max` plus one video plus its
soundtrack:

| generated | sequence | result |
|---|---|---|
| 124 frames | 78,019 | **success**, 740s, peak 21,938 MiB |
| 345 frames | 182,092 | **OOM** at step 4 of 16, 21.05 GiB allocated |

**The video-reference arms ship 362 frames as of 2026-08-16**, not the 345
these rows were measured at. `REF_VIDEO_LENGTH` was deleted rather than raised
-- a safe length here depends on reference count, kind, duration, canvas and
upscale, so no constant is right twice. Treat the table as the shape of the
cliff, not as clearance, and read preflight before any reference render.

The failure is graceful and worth recognising: Sol-Attn's kernel OOMed and
fell back, then sage's OOMed and fell back, then ComfyUI's own SDPA OOMed.
Three clean degradations, each logged. There was simply no room.

Note the margin at the successful run — 21,938 of 24,564 MiB, about 2.6 GB
spare. Reference video is the most expensive input in the model.

**Budget by pixel area, not by count.** The same clip at 640x360 costs a third
of what it costs at 960x544.

---

## The vendor image path, stage by stage

Read from source 2026-08-21 against `coderef/sglang` at commit `a41da991c8`,
alongside ComfyUI's `comfy_extras/nodes_minimax_h3.py`,
`comfy/text_encoders/minimax.py` and `comfy/ldm/minimax/model.py`. **This
section covers the image path only.** The video and audio paths have their own
vendor-side attributions, re-derived 2026-08-21 against sglang, diffusers and
DiffSynth-Studio in the sections that own them.

The headline: the two implementations are the same pipeline written twice, and
the largest divergence is **how large a reference image is when it reaches the
two towers.** Downstream of the resize the presentation, patch geometry,
packing, condition timestep and condition-noise recipe all match; the VAE
encode does not. **Corrected 2026-08-21** — this paragraph said "everything
downstream of the resize matches" until the posterior row below was added.

| stage | sglang | ComfyUI | same? |
|---|---|---|---|
| admission | validates `minimax_h3.request/v1`, resolves the release partition from `model_index.json._minimax_h3`, refuses a task outside it (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py:60-135`) | no partition concept; any checkpoint takes any graph | **no**, see below |
| geometry | resolved before queueing and frozen; the runtime is forbidden from recomputing it from the plan (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/prequeue.py:303-315`) | computed inline in the node at execute time | no consequence found |
| sizing | 2048 short edge, upscale included, nearest-32 per axis, ratio 1:4..4:1 enforced, no area cap (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:125-203`) | `min(1.0, ...)` in both modes; default `match` targets the generation area | **no — the whole divergence** |
| resampler | PIL LANCZOS (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:181-203`) | `common_upscale(..., "lanczos")` on the tensor (`comfy_extras/nodes_minimax_h3.py:64-68`) | same intent |
| orientation | `ImageOps.exif_transpose` on open (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:886-887`) | none — a tensor arrives already decoded | not applicable |
| one image, two towers | the identical prepared image feeds Qwen `pixel_values` and the visual-condition tokenizer, cached on the batch (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:840-846`) | the same `resized` tensor goes to `vae.encode` and into `ref_items` (`comfy_extras/nodes_minimax_h3.py:304-306`) | **yes, until 3.0625:1** — past that the Qwen ceiling shrinks one branch and not the other; measured below |
| VAE encode | the released `DiagonalGaussian` is **sampled** under a pinned seed 42, not averaged (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/keyframe_encoding.py:7-30`, and `MINIMAX_H3_REFERENCE_VIDEO_ENCODE_SEED` at `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:613`) | the posterior **mean**, `torch.chunk(moments, 2, dim=1)[0]`, never sampled (`comfy/ldm/minimax/vae.py:685`) | **no**, found 2026-08-21 |
| patchify | delegated to the HF `image_processor`, which reads the release's own `processor/preprocessor_config.json` | reimplemented, with the geometry right and two bounds left on the shared helper's signature defaults (`comfy/text_encoders/minimax.py:35-66`, `comfy/text_encoders/qwen3vl.py:62-68`) | **algorithm yes, constants no** |
| presentation | `<Picture i>: ` then a vision block, no chat template (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/presentation.py:104`) | the same string, the same place (`comfy/text_encoders/minimax.py:164`) | **yes** |
| packing | ref blocks in request order, target timeline starts past their spans (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py:274-320`) | `_ref_t_span` does the same (`comfy/ldm/minimax/model.py:335-338`) | **yes** |
| condition timestep | 0.999, applied as `max(t_video, aug)` (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py:24`) | 0.999, applied as `max(t_v, vis_aug)` (`comfy/ldm/minimax/model.py:32`) | **yes** |
| condition noise | the same 0.999 is the mixing weight: per-condition CPU generator, `aug * clean + (1 - aug) * noise` (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/condition_noise.py:93-117`) | the same recipe, the same constant, a fresh generator per condition (`comfy/ldm/minimax/model.py:499-511`) | **recipe yes, draw no** -- corrected 2026-08-28, see the note below |

**Four consequences worth carrying away.**

**The two VAEs disagree in opposite directions, and only one of them is a
divergence.** Recorded 2026-08-21 after an outside review pointed at the audio
half, which this section had not covered. sglang samples the **video** VAE
posterior at a pinned seed and takes the **audio** VAE posterior *mean*
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/audio_encoding.py:83`, whose docstring says "audio VAE posterior
mean" in as many words). ComfyUI takes the mean for both. So ComfyUI diverges
on every reference image and reference video and **matches the release on
reference audio** -- and it matches on precision too, since `comfy/sd.py`
pins the H3 audio VAE to `working_dtypes = [torch.float32]`. The asymmetry is
in the release, not in ComfyUI: the audio VAE's own `logs_proj` ships in the
checkpoint and is unused at inference on both sides.

**The condition latent is drawn differently, and this is separate from the
condition noise below.** Read on both sides 2026-08-21. sglang samples the
released posterior with a seed pinned at 42 for keyframes and for reference
video; ComfyUI takes the mean and has no sampling path at all. So the latent
that reaches the DiT differs before any noise is mixed in, on every reference
image, reference video and keyframe. Two caveats before anyone measures it.
Our video VAE is an fp16 cast of the release fp32, so a parity test carries
precision and mean-versus-sample as two variables at once and has to separate
them. And a rendered clip cannot answer it — `CLAUDE.md`'s different-sample
rule applies, so the comparison has to be made at the latent, not the output.

**The condition-noise recipe matches and the DRAW does not, found 2026-08-28.**
The table row said a flat yes until then. The mixing weight, the constant and
the per-condition generator are the same on both sides; what differs is where
the noise is drawn. sglang draws in latent space at the target's own length
plus its conditions and prefix-slices before patchify
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/condition_noise.py:94-119`);
ComfyUI draws in row space after patchify, at the condition's own size
(`comfy/ldm/minimax/model.py:505-511`). Two independent reasons the tensors
differ at one seed.

**Read this as a difference, not a defect.** *Source read, not measured.* The
recipe being identical is the part that governs behaviour; a different draw at
one seed is a different sample, which `CLAUDE.md`'s different-sample rule says
cannot be ranked by looking at two clips. Nothing here says sglang's draw is
the correct one — **it is the vendor's serving path, not a specification**, and
several of its choices follow from hardware we do not have. This box is a
24 GB card with 128 GB of host memory streaming blocks through CPU RAM to fit a
model that would not otherwise fit; a serving stack sized for datacentre parts
makes different trades and some of them do not transfer. Where a mechanism
*does* transfer it is worth knowing, which is why the row is kept rather than
dropped. What would settle whether the draw matters at all is a latent-space
comparison at one seed, not a render.

**The smart-resize policies do not match the release, found 2026-08-21.** The
patch geometry does: `patch_size=16`, `temporal_patch_size=2`, `merge_size=2`
and the 0.5 mean/std are the same on both sides, and ComfyUI passes 16
explicitly rather than inheriting Qwen2-VL's 14
(`comfy/text_encoders/qwen3vl.py:62-68`). The **pixel bounds** do not. The
release ships still bounds in `processor/preprocessor_config.json` and video
bounds in `processor/video_preprocessor_config.json`, both as
`size.shortest_edge` / `size.longest_edge`. ComfyUI reads neither file: it
leaves `process_qwen2vl_images` on the signature defaults of the shared
Qwen2-VL helper, and `process_video_block` carries the same pair for each
two-frame block. The released video processor instead applies its pixel budget
over the whole sampled clip. The model is Qwen3-VL-32B on both sides — this is
one shared helper's defaults reaching a model it was not written for, not a
wrong model.

| bound | H3 release | ComfyUI |
|---|---|---|
| image min pixels | 65,536 | 3,136 |
| image max pixels | 16,777,216 | 12,845,056 |
| video min pixels | 4,096 | 3,136 |
| video max pixels | 25,165,824 | 12,845,056 |

The still-image edges bite in opposite directions. **Below:** the release
*enlarges* anything under 65,536 pixels to reach that floor; ComfyUI's floor is
twenty times lower, so a small reference is under-tokenized rather than raised.
That is a second, independent way a small reference arrives smaller than the
release intends, on top of the `min(1.0, ...)` clamp — and unlike the clamp,
`MiniMaxH3ReferenceFit` does not close it, because it operates before the
tokenizer.

Reference video is a different failure shape. An individual canvas-sized frame
is below either numeric maximum, but the release's 25,165,824-pixel maximum is
clip-wide and becomes duration-sensitive. Native ComfyUI applies 12,845,056 to
each two-frame block independently, so it never enforces that whole-clip row
budget. The executed boundary is source-dependent: it starts at legal H3
lengths of 311 frames for a 1344x768 input, but is outside the legal range for
the measured 960x544 input. The locally shipped `encoder` policy applies the
loaded encoder's duration-aware Qwen stage, read off the CLIP's stamped
contract (a CLIP that declares nothing resolves to native); `comfy` retains
native behavior and `release` applies both vendor video stages.

**Above, and this is measured rather than derived.**
`bench/measure_qwen_bounds_bite.py` calls the real `process_qwen2vl_images`
with the arguments `comfy/text_encoders/qwen3vl.py:65` passes and reports the
grid it gets back; record in
[`bench/results/2026-08-21_qwen_bounds_bite.json`](../bench/results/2026-08-21_qwen_bounds_bite.json).
References prepared to a 2048 short edge:

| prepared | what Qwen sees | release |
|---|---|---|
| 6144x2048 (3:1) | 6144x2048, untouched | untouched |
| 6272x2048 | 6272x2048, untouched | untouched |
| **6656x2048 (3.25:1)** | **6432x1984, shrunk** | untouched |
| 7168x2048 (3.5:1) | 6688x1888, shrunk | untouched |
| 8192x2048 (4:1) | 7168x1792, shrunk | untouched |

So the ceiling starts biting between 3:1 and 3.25:1 — the crossing is at
`12,845,056 / 2048² = 3.0625` — and the release carries the same image
untouched all the way to 4:1, which is the widest ratio the reference resize
accepts at all. **The consequence is worse than the shrink itself: it breaks
the one-image-two-towers row above.** The VAE still receives the full
2048-short-edge tensor while Qwen silently receives a smaller one, so past
3.0625:1 the two towers are no longer looking at the same picture, and nothing
says so.

The same script carries the arm that says where this *cannot* happen. Every
legal H3 canvas — the keyframe case — comes back untouched, because a canvas is
always a multiple of 32, which is exactly the `patch_size * merge_size` factor
the helper rounds to, and every legal canvas sits inside both implementations'
floors and ceilings. So the resize never runs on a keyframe and the
bilinear-against-bicubic difference never fires there either. That arm is the
control: if a canvas ever is resized, the inert claim in
[`official_weights_metadata.md`](research/official_weights_metadata.md) is
wrong and the script goes loud.

**Being seen in patches is still not a mechanism for anything.** The bounds
above are a sizing difference wearing the tokenizer's clothes: they decide how
big the image is before it is cut up, not how it is cut up. Patchified images
are what the model was conditioned on in the first place, on every
implementation, so "Qwen sees it in patches rather than natively" describes the
training distribution rather than a departure from it. What can cost you is
arriving at the tokenizer a different size than the release would have chosen —
which is the resize, and now also the floor.

The partition refusal is a fact about the release, and **the release itself is
where it comes from** — confirmed 2026-08-21 by reading the official
weights rather than sglang's reader. MiniMax ships two partitioned entry
points beside the shared components, and each carries a `_minimax_h3` block in
its own `model_index.json`: the fl2va one declares `tasks: ["t2va", "fl2va"]`,
the ref2va one declares `tasks: ["ref2va"]` and nothing else. So "ref2va is not
a text-to-video model" is the vendor's own metadata, not an inference from a
serving implementation. sglang reads that `partition` field and
raises when the requested task does not belong to it, which is why a `t2va`
request against the `ref2va` partition is refused rather than served badly.
ComfyUI has no such gate: a plain t2v graph will load `ref2va` and render. Read
a `ref2va` loss at plain t2v as **not a t2v model by its own release
metadata**, not as a defect.

**The condition rows are noised on both sides, and the one difference is the
shape of the draw.** Worth stating because the timestep row above invites the
opposite reading. sglang and DiffSynth-Studio both draw the noise in latent
space over the full conditioned length, slice it, then patchify
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/condition_noise.py:93-113`); ComfyUI draws directly in patchified row
space (`comfy/ldm/minimax/model.py:507`). Same distribution, different
realisation for the same seed, and two of three implementations agree against
ComfyUI. At the shipped aug the noise carries a thousandth of the weight, so
this changes which sample you get and is not an argument about fidelity. It
would matter if anyone drove the aug low, and nothing ships that: ComfyUI reads
`minimax_visual_cond_noise_aug` and `minimax_audio_cond_noise_aug` out of the
conditioning at the payload boundary (`comfy/model_base.py:2175-2179`) and **no
shipped node sets either**, so the knob is reachable only from a custom node.

---

## Sizing a reference image: two knobs, not one

The single most confused thing on this page, and the confusion was in this
document until 2026-08-16. `ref_image_size` on `MiniMaxH3ReferenceToVideo` and
`allow_upscale` on `MiniMaxH3ReferenceFit` are **orthogonal**. Neither is a
version of the other.

### What each one actually does

`comfy_extras/nodes_minimax_h3.py:297-301`, read from source:

```python
if ref_image_size == "match":
    scale = min(1.0, math.sqrt((width * height) / (w * h)))
else:  # "max"
    scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))   # 2048
```

**Both branches clamp with `min(1.0, ...)`. Core never upscales in either
mode.** `max` changes *which ceiling* sizes a reference down — the 2048 short
edge instead of the generation's pixel area — not the direction. Core's own
socket tooltip says so: "downscaled to 2048 short edge if larger, **never
upscaled**".

`reference_fit.py:181` is the only thing in the stack that drops that clamp:

```python
scale = full if allow_upscale else min(1.0, full)
```

So: **`ref_image_size` picks the ceiling; `allow_upscale` decides whether a
small reference is raised to it.** You need `max` *and* `allow_upscale=True` to
condition at 2048.

### Scope: one is global, one is per reference

| | `ref_image_size` | `allow_upscale` |
|---|---|---|
| lives on | `MiniMaxH3ReferenceToVideo` | `MiniMaxH3AppendRefImage` |
| applies to | **all image references at once** | **one reference** |
| touches video refs | no | no |
| touches audio refs | no | no |

`ref_image_size` is read in exactly one loop, the `ref_images` one. Video
references never see it — they take `adapt_canvas` from their own aspect ratio
(`:315-319`) — and audio has no spatial sizing at all.

`MiniMaxH3AppendRefImage` takes **one image per node** and warns if you feed it
a batch, so upscaling is decided per reference. Five references means five
nodes and five independent decisions. The retired `MiniMaxH3ReferenceFit` did
the same and still does, for saved graphs that wire it.

### The four combinations — PRE-FOLD WIRING ONLY

**Scope, added 2026-08-27 because this table was applied to the shipped path
and produced a wrong conclusion.** Everything in this subsection describes
**two** nodes: `MiniMaxH3ReferenceFit`'s `allow_upscale` feeding **core's**
`MiniMaxH3ReferenceToVideo.ref_image_size`. Two nodes means two resizes, which
is where row two's wasted work comes from. **No shipped graph is wired this
way** — see the subsection below for what the typed path does. The table is
kept because saved graphs outside this repo still wire `MiniMaxH3ReferenceFit`,
which is the only reason that node is still registered.

| ReferenceFit | core conditioning node | what happens |
|---|---|---|
| `allow_upscale=True` | `max` | 2048 survives. **The working combination.** |
| `allow_upscale=True` | `match` | upscale **undone** — core re-shrinks to the generation's pixel area, and you paid two lanczos resamples for nothing |
| `allow_upscale=False` | `max` | both no-ops for anything under 2048 |
| `allow_upscale=False` | `match` | reference sized to the generation's pixel area |

Row two is why the node inspects its consumer at run time
(`reference_fit.py:199-206`) and warns. Row three is the state **10 of the 18
graphs that wire ReferenceFit are in**, deliberately — `REF_VIDEO_BUDGET`
holds `allow_upscale=False` to fit 24 GB. As of 2026-08-16 the node says "NO
CHANGE" in the log when it lands there, because its presence in a graph
otherwise reads as "the references were fitted".

### What the shipped path does instead, and why row two cannot happen on it

`MiniMaxH3AppendRefImage` carries `size_policy`, `short_edge` and
`allow_upscale` itself, and `MiniMaxH3ReferenceConditioning` performs **one**
resize with the canvas in scope. `reference_fit.py`'s own docstring records the
fold: "On the typed reference path this node is no longer needed."

So the two knobs above are no longer on two nodes, and the `match` branch does
not read `allow_upscale` at all
([`reference_geometry.py::fit_reference_image`](../reference_geometry.py)):

```python
if size_policy == "match":
    # allow_upscale is never read here
    scale = min(1.0, math.sqrt((canvas_w * canvas_h) / (source_w * source_h)))
else:
    full  = short_edge / min(source_w, source_h)
    scale = full if allow_upscale else min(1.0, full)
```

`match` + `allow_upscale=True` on the shipped path is therefore **not a double
resample**. One resize either way, and no wasted lanczos pass.

**And since 2026-08-27 the combination is not expressible.** `size_policy` is a
`DynamicCombo`: `short_edge` and `allow_upscale` are nested under the `max`
branch, so they do not exist when `match` is selected. The node used to carry
them as flat widgets and log a warning that they were not read — visible only
after a render was queued. The warning is gone because its subject is
unreachable.

**The reason to prefer `max` for a vendor-matching arm is the one this page
already gives above and not row two**: `match` sizes from the generation's
pixel area, so its ceiling moves with the canvas, where sglang's is a fixed
2048 short edge with no area cap. That is a divergence in a different
direction, and it is what makes `match` the wrong policy for parity — not a
resample it does not perform.

### Worked examples

Five image references, `ref_image_size='max'` on the conditioning node
throughout, `short_edge=2048`. Rows are `(w/32) x (h/32)` after the round-to-32
— the VAE compresses by 16 and the DiT patchifies 2x2 on top. Computed from the
formulas above, not measured.

| source | short edge | upscale off | rows | upscale on | rows | factor |
|---|---|---|---|---|---|---|
| 1024x512 | 512 | 1024x512 | 512 | 4096x2048 | 8,192 | **16.00x** |
| 960x1280 | 960 | 960x1280 | 1,200 | 2048x2720 | 5,440 | 4.53x |
| 2048x2612 | 2048 | 2048x2624 | 5,248 | 2048x2624 | 5,248 | **1.00x** |
| 768x512 | 512 | 768x512 | 384 | 3072x2048 | 6,144 | **16.00x** |
| 1920x1080 | 1080 | 1920x1088 | 2,040 | 3648x2048 | 7,296 | 3.58x |

`2048x2612` is already at the target, so `allow_upscale` changes nothing —
`full = 2048/2048 = 1.0` in both modes. That is the second "no change" case the
node reports.

Mixing them, same five references:

| scenario | reference rows | vs baseline | with the text twin |
|---|---|---|---|
| none upscaled | 9,384 | 1.00x | ~18,768 |
| **only the two 512-short-edge** (1024x512, 768x512) | 22,824 | 2.43x | ~45,648 |
| only the two largest (960x1280, 1920x1080) | 18,880 | 2.01x | ~37,760 |
| all upscaled | 32,320 | 3.44x | ~64,640 |

**The counterintuitive one, and the reason this table is here: the smaller the
reference, the more upscaling costs it.** Rows are quadratic in the scale
factor and the scale factor is `2048 / short_edge`, so a 512-short-edge image
pays 16x and a 1080 one pays 3.58x. Upscaling the two *smallest* references
above adds **13,440 rows**; upscaling the two *largest* adds **9,496**. The
instinct that big references are the expensive ones is backwards — big
references are already close to the ceiling, and the ceiling is where everything
ends up.

The "text twin" column is not padding. Reference images cost again in the text
segment, at 75-160 rows *above* the reference segment itself (measured, see the
count ladder above), because the conditioner's vision blocks read the resized
image too. So all five upscaled is roughly **64,600 rows before a single frame
of video** — against 37,296 for an entire 124-frame render at 1344x768.

Nothing here says the 2048 version looks better. That is
`docs/open_experiments.md` #1, still unmeasured, and its own entry notes that
2048 "is a good reason to offer it and a weaker reason to default to it,
because upscaling adds tokens rather than detail."

### Both stages on three real sizes, and what the v2 decision changes

The table above is the DiT half. A reference is sized twice, and the second
stage is the encoder's own processor bounds, which can only shrink what Qwen
sees. Computed 2026-08-25 through the shipped functions rather than the
formulas: `reference_geometry.fit_reference_image` for stage one,
`reference_geometry.qwen_image_size` under the loaded encoder's stamped
contract for stage two (`bench/check_reference_runtime.py` holds both to the
installed processors). "v1" is the current W4 artifact's 200,704--301,056-pixel
snapshot; "v2" is the release declaration, 65,536--16,777,216, which the v2
candidate is calibrated and served at.

| source | choice | stage 1 (VAE and Qwen input) | DiT rows | Qwen tokens, v1 | Qwen tokens, v2 |
|---|---|---|---:|---:|---:|
| 640x480 | A: `max`, no upscale | 640x480 | 300 | 266 | 300 |
| | B: 2048 short edge, upscale | 2720x2048 | 5,440 | 266 | 5,440 |
| 1920x1080 | A | 1920x1088 | 2,040 | 264 | 2,040 |
| | B | 3648x2048 | 7,296 | 264 | 7,296 |
| 3024x4032 | A | 2048x2720 | 5,440 | 266 | 5,440 |
| | B | 2048x2720 | 5,440 | 266 | 5,440 |

Four things to read off it:

- **A and B differ only for sources below a 2048 short edge.** The portrait
  is already past it, so both choices cap it at 2048 and the rows are
  identical; the crop and the 1080p photo are where the choice exists.
- **Under v1, A versus B changes only the DiT.** Stage two crushes Qwen to
  about 265 tokens either way, so B buys 5,440 DiT rows for the crop instead
  of 300 and the conditioner never sees the extra pixels. That asymmetry is
  why no-upscale was the sound default while v1 was the encoder.
- **Under v2, B multiplies what Qwen sees.** The crop goes from 300 to 5,440
  Qwen tokens, every one interpolated from 640x480; the photo from 2,040 to
  7,296. A reference pays twice, as DiT rows and as Qwen tokens in the text
  segment, and both sit inside Sol-Attn's exact sink
  ([`SOLATTN.md`](SOLATTN.md)), attended every step.
- **The choice is distribution against information.** A feeds the encoder
  real pixels only and is far cheaper. B feeds it the geometry the vendor's
  serving pipeline produces (the sglang rule at the top of this section),
  which is what the DiT was trained to read layer-50 states of.

**OWNER-DECISION, 2026-08-25:** v2 calibrates reference stills at B, the
vendor's 2048 upscale, as the primary policy, cost accepted; `max` with no
upscale stays as a comparison stratum. Recorded in
[`canonical/active_plan.md`](research/qwen3-vl-special-tokens-post-training/canonical/active_plan.md).
The consequence for graphs: a v2 parity graph needs the upscale at stage one
(`allow_upscale=True` on the fit or append) and `video_policy=release` for
video references, in addition to the encoder contract the loader stamps. On
2026-08-25 the shipped graphs are 6 upscale-on to 28 off and 39 of 40 on
`video_policy=encoder`, so most of them would run v2 at A geometry: the AWQ
scales chosen from B-sized activations applied to A-sized inputs. That is the
same class of mismatch measured for v1 in the other direction (vision cosine
0.966 at its calibration geometry, 0.832 served wider;
[`2026-08-24_layer50_processor_policy_benchmark.md`](research/qwen3-vl-special-tokens-post-training/canonical/2026-08-24_layer50_processor_policy_benchmark.md)),
not yet measured in this one. Which graphs flip, and when, is the owner's
call at v2 acceptance. The paragraph above this one still stands: none of
this says the 2048 version looks better; it says which distribution the
encoder was calibrated on.

### A third knob, 2026-08-25: a Qwen view of its own

`MiniMaxH3AppendRefImage.qwen_short_edge` (default 0) gives the text encoder a
view of the reference that the video VAE does not encode. With 0, Qwen sees
the same tensor the VAE encodes, which is every graph built before the knob
existed. With N, the conditioner is shown the *source* scaled so its shorter
side reaches N (nearest 32, one Lanczos resample, upscaling allowed, so a 4k
source comes down to N as well), while the VAE keeps the stage-one view chosen
by `size_policy` / `short_edge` / `allow_upscale`. Under `image_policy` of
`encoder` or `release` the stage-two bounds are pre-applied to the Qwen view
alone; the VAE view is no longer clamped when a Qwen view exists.

Why this breaks no contract is section 1b of
[`h3_conditioning_end_to_end.md`](h3_conditioning_end_to_end.md): nothing
indexes a Qwen token against a latent patch, the deployed v1 path has always
run VAE-fine / Qwen-coarse, and video is 2 fps pairs against 24 fps latents by
design. What it changes is the cost split in the table above: only the Qwen
column grows. A 640x480 reference at `qwen_short_edge=2048` costs the same 300
reference-latent rows as choice A and the same 5,440 Qwen tokens as choice B,
which is the B arm of the reference-view ablation (A: no upscale; B: Qwen-only
2048; C: full parity). Whether B helps is unmeasured and is the owner's blind
matched-seed comparison to judge after v2 lands.

**The caveat, and which regime you are in decides everything.** The loaded
encoder's own processor applies its bounds afterwards, so this knob is only ever
worth as much as the selected artifact's still-image budget allows. There are
three regimes and they do not agree:

| encoder | still bounds | what `qwen_short_edge` does |
|---|---|---|
| v1 W4 snapshot | 200,704..301,056 | **exactly inert** on every non-square aspect -- 512, 1024 and 2048 all arrive as 264 merged tokens at 16:9 |
| v2 W4 snapshot | the release's own | live, 448..7,296 merged tokens across that range |
| **`ENCODER_INT8`, what ships** | **core's `process_qwen2vl_images` defaults, 3,136..12,845,056** | **live, and the widest of the three** |

**Corrected 2026-08-28: this section said the shipped encoder is v2. It is
not.** `h3_config.MODELS["clip"]` is `ENCODER_INT8`
(`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`) and every `CLIPLoader` in
the shipped graphs loads it; the v2 lane was closed rather than adopted. The
conclusion that the knob is load-bearing survives the correction, but **not for
the reason given here** -- it is live because INT8 loads through core's
`CLIPLoader`, which stamps no `_h3_encoder_contract`, so no snapshot's bounds
are installed and core's own defaults apply, 43x wider than v1's.
`h3_config.py`'s note on `ENCODER_INT8` owns this and is the current one. Read the
budget with `h3_awq_encoder.py::source_image_pixel_bounds`, never from prose;
`bench/preflight_graph.py` prices both views per reference and says on the
line when the Qwen view was clamped and by whose bounds. Controlled by
`bench/check_reference_runtime.py::qwen_view_is_separate_from_the_vae_view`
and `preflight_prices_the_two_views`; the red harness's M9 feeds the Qwen view
to the VAE and must go red.

### Should video and audio have fit nodes too?

Both have the same kind of divergence from the reference pipeline. The answers
go opposite ways.

**Audio: yes, at this repo's typed boundary.** Native ComfyUI encodes the whole
waveform (`_encode_ref_audio`, no truncation on either material path) where the
reference pipeline cuts it to generated duration. That costs 80 rows per second
of excess. The first local handling used explicit `TrimAudioDuration` nodes in
2026-08-22; the typed migration replaced them with one compiler-derived cap,
which cannot drift from `length`.

The same boundary duplicates mono to stereo and refuses more than two channels.
Native `_encode_ref_audio` still raises on mono, so both duration and channel
handling must remain described as local behavior. Preflight reports native
socket graphs separately, including "unreadable" when ffprobe cannot answer.

**Video: full release parity remains opt-in; the encoder-aware hybrid is now
the generated default.** The divergence is the same shape as the image one —
native ComfyUI never upscales, where the release puts the clip on the full
canvas rule — but video is the most expensive reference input.
`MiniMaxH3ReferenceConditioning.video_policy=encoder` keeps the native
no-upscale VAE view while applying the custom encoder's source-config,
duration-aware processor to the raw 2 fps Qwen samples. `comfy` remains the
native preprocessing control; `release` is the explicit full-parity experiment.

`release` owns two inseparable stages. It upscales the full-rate view to
`adapt_canvas` for the video VAE, then independently sends the raw 2 fps Qwen
samples through the release's `Qwen3VLVideoProcessor`. Above the processor's
duration boundary, Qwen may shrink its view while the VAE keeps the full
canvas. Exposing “upscale only” would overshoot the release's Qwen rows, so it
is deliberately not an option.

This is handling in our custom node, not native ComfyUI. Core's
`MiniMaxH3ReferenceToVideo` remains no-upscale and duration-unaware. The
39-frame live smoke proves the local policy executes; it does not establish
that upscaling improves a clip. `MiniMaxH3ReferenceVideoFit` remains useful for
reporting or deliberate downscaling on native-compatible paths, not as a
substitute for the atomic release policy.

---

## Labels: the tokenizer decides, not the prompt

Native socket references are emitted in a fixed order with a **separate
1-based counter per type**:

1. images, as `<Picture i>`
2. then each video: its paired soundtrack's `<Audio j>` **immediately before**
   its `<Video k>`
3. then standalone audio, continuing the `<Audio j>` count

So one video with sound plus one standalone clip reads `<Audio 1>`,
`<Video 1>`, `<Audio 2>` — the soundtrack takes the first audio ordinal and the
standalone clip is second, while the video is `<Video 1>` either way. This is
easy to get wrong by hand, and getting it wrong is silent: the render succeeds
and quietly ignores an instruction about something that is not there.

`bench/check_ref_prompt_labels.py` asserts every shipped graph's prompt names
exactly what its graph wires, in this numbering. It also catches the reverse —
a wired reference the prompt never mentions, which still costs its rows on
every step and is the most expensive way to say nothing.

This repo's typed surface replaces fixed grouping with list position and suffix
pairing with ownership. The label counters and sounded-video rule are unchanged:
an owned soundtrack still emits its `<Audio j>` immediately before its
`<Video k>`. Generated workflows preserve the native group order during this
migration; arbitrary order is available only when a graph explicitly builds it.

`<Video N>` and `<Audio N>` are numbered independently, and an ordinary
reference video does not create an `<Audio N>` merely because the file has
sound. Its native soundtrack socket or typed soundtrack input has to be wired.

**A silent clip must not have its soundtrack pulled.** VHS raises "failed to
extract audio" when its audio output is pulled on a video with no audio
stream, and the render dies at execution having validated cleanly.

---

## A label is a bare ordinal. You have to say what it is.

**Measured 2026-08-16**, paired render, one variable —
`docs/prompt_length_experiment.md` for the full pre-registration and verdict.

The tokenizer emits `<Picture 1>`, `<Picture 2>`, `<Video 1>` and nothing else.
There is no socket, flag or payload field carrying what a reference *is*, what
it is *for*, or which subject it belongs to — verified against
`comfy/ldm/minimax/model.py`. Three images wired to one subject and three
images wired to three subjects are **identical graphs**. The model only knows
which you meant because the prompt says so.

So the prompt is not a description of the output. It is the only place the
inputs acquire meaning.

### The authority order, so far

Two n=1 results on this model in this pipeline, in the same direction. Neither
is in any guide.

1. **`subject_definitions` beats `retention_analysis`.** A brunette reference
   described as blonde in the definitions rendered blonde, despite
   `fully_preserved` on the retention line. Owner's observation.
2. **A specific `detailed_description` beats `subject_definitions`.** Both arms
   of the length experiment carried a byte-identical definition claiming
   `<Subject 2>` had "architecture", against a mountain-lake reference with no
   buildings in it. The arm whose description said nothing about the
   environment rendered the subject **inside a timber veranda with a chalet
   beside it**. The arm whose description named the lake, the reflection, the
   meadow and the conifers produced **no structure at all**.

Read together: **a wrong word in `subject_definitions` is load-bearing exactly
to the extent that nothing downstream contradicts it.** Silence downstream is
not neutral — it leaves the definition in charge.

### What follows for writing one

- **Say what each reference is**, at least at the level of a noun and its
  salient attributes. "the environment in `<Picture 2>`" is thin; the model has
  to infer everything from the pixels and whatever adjectives you supplied.
- **Do not assert an attribute you have not looked at.** The generic template
  that shipped on every image-reference arm until 2026-08-16 said "architecture,
  palette, and lighting" for whatever image happened to be wired. It now says
  "setting", which is true of any environment. `_ref_prompt()` cannot see the
  image; a person writing by hand can and should.
- **Describe the environment in `detailed_description` even when a reference
  supplies it.** This is the counter-intuitive one. It feels redundant — the
  reference is *right there* — and it is the difference between the two clips
  above.
- **A reference wired but never described costs its rows on every step and
  says nothing.** `bench/check_ref_prompt_labels.py` catches the unnamed case;
  `bench/preflight_graph.py` also warns when a defined label is never cited in
  `detailed_description`, which is the guide's actual requirement
  (`coderef/MiniMax-H3/skills/h3-prompt-writing/references/ref-en.txt:231`)
  and the half `subjects_resolve` did not implement.

### One caution on the experiment behind this

The length arm had **no working control**. Both prompts carried the identical
camera sentence, which was supposed to bound seed noise; the long arm executed
the move and the short one did not, because the long arm elaborated it with
consequences. You cannot hold one sentence constant inside a prompt whose
length you are varying — the model reads it in context, not as a string. So the
direction above is credible (four results agree, and the architecture one is
binary rather than aesthetic) and the magnitude is not bounded.

## Prompt structure

**These six sections are the reference format only, and that is not obvious.**
Found 2026-08-21. The guide ships two output formats and this page had only
ever described one. `coderef/MiniMax-H3/skills/h3-prompt-writing/references/ref-en.txt:311-337` is the six-section format below,
for ref2va. `coderef/MiniMax-H3/skills/h3-prompt-writing/references/base-en.txt:39-43` is what t2va, i2va, fl2va and l2va use, and
it is **three** fields:

```
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

No `subject_definitions`, no `summary`, no `retention_analysis`, and
`integrated_multimodal_description` where the reference format says
`detailed_description`. `<Subject N>` labels are reference-format too -- the
base guide describes people inline and carries only the `(S1)` speaker ids.
Keyframe tasks prepend one optional line above the three fields saying how each
`<Picture N>` maps to a second mark in the target.

Both prompt instruments now select the guide from the graph itself and support
native sockets plus typed append chains. `bench/preflight_graph.py` and
`bench/check_prompt_guide_conformance.py` grade ref2va against this six-section
format and base/keyframe tasks against base-en's three fields.

Six sections, in this order:

```
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:
```

`<Subject N>` is reusable visible **content**. `<Picture N>` / `<Video N>` /
`<Audio N>` identify the **assets**. If a person, object, scene or action from
a reference video is reused as visible content, **it still belongs under
`<Subject N>`** — `<Video N>` identifies the source and does not replace
subject labels. One subject may be defined by several assets, naming what each
provides.

### Markers, and they do not interchange

| visible content | audio |
|---|---|
| `fully_preserved` | `fully_copy` |
| `partially_preserved` | `partially_copy` |
| `attribute_transfer` | `reference` |
| `weak_reference` | `weak_reference` |

Only `weak_reference` appears in both sets. The label check verifies labels
exist; **it does not verify you picked a sensible marker**, so that part is on
you.

---

## The relationships, and how to ask for them

Which sockets you wire is mechanical. **What the prompt asks those labels to
do is the axis that changes the output.** The official guide names three
whole-video relationships plus a subject-sourcing rule that yields a fourth.

### Edit a source video (the closest thing to inpainting)

**There is no mask.** This is whole-frame regeneration conditioned on the
source; what holds the untouched parts still is `retention_analysis` saying
precisely what survives.

```
subject_definitions:
<Subject 1> is the person in <Video 1>, whose face, build, and position in frame are kept in the target video.
<Subject 2> is a bright red waxed-cotton jacket that replaces the garment <Subject 1> wears in <Video 1>.
<Video 1> is the source video for the target video edit.

retention_analysis:
<Subject 1> (appears in [Shot 1]): partially_preserved - face, build, posture, and motion are retained from <Video 1>; the garment changes.
<Subject 2> (appears in [Shot 1]): attribute_transfer - the red jacket replaces the original garment on <Subject 1>.
<Video 1> (source video for the edit): partially_preserved - framing, camera movement, and shot timing are kept; only what is named above changes.
```

`partially_preserved` is the marker meaning "keep this, except".
`fully_preserved` asks for a copy and leaves the edit nowhere to happen;
`weak_reference` throws away the framing you are trying to keep.

Graph: `h3_ref_video_edit.json`.

### Replace a character, keeping the video as the plate

The same sockets as the edit above, pointed at the opposite question: there
the person stays and the garment changes, here the person is the only thing
that changes. `<Video 1>` is the *plate*; `<Picture 1>` is the new identity.

**Speech in a reused soundtrack came out damaged on three of four renders,
2026-08-22, and the prompt is NOT established as the cause.** Three prompts --
the structured arm below, the concise twin, and a third imperative arm carried
from a community write-up -- were rendered on identical references, canvas
(1152x768) and length (124 frames). Four renders landed before the batch was
stopped: structured at two seeds, concise and imperative at one each. The
owner, on playback, found exactly one of the four clean, and it was a
structured render -- but **the other structured render was not clean**, so the
split does not follow the prompt.

**What that supports and what it does not.** It supports the owner's call to
delete the imperative arm, which was made and is recorded in the changelog. It
does **not** support "the six sections protect the speech": one clean render
of four, with the only within-arm pair disagreeing with itself, is a draw from
a distribution and says nothing about the knob -- `CLAUDE.md`'s
different-sample rule applies to prompts as much as to numbers. The concise
twin stays shipped because nothing here refuted it either.

**Two things were ruled out afterwards and one hypothesis is still open.**
The source clip is **25 fps** (489 frames / 19.560s, `r_frame_rate=25/1`),
which is exactly the rate the fps gap in
[`comfyui_vendor_gaps.md`](comfyui_vendor_gaps.md) is about -- but the
workaround is wired: these graphs set `force_rate=24`, which VHS applies as an
ffmpeg `fps=fps=24` filter, preserving wall-clock. And the audio window VHS
hands out is `frame_load_cap / force_rate` seconds from the same origin, so
the frame window and the soundtrack window cover the same span of the source.
Neither desync is present. **Span expansion is ruled out for that batch too**, and it
was the appealing explanation: the runs were patched to `frame_load_cap=124`,
so VHS handed out 5.167s of soundtrack against a 5.167s visual reference and
the cursor formula above had nothing to expand. Checked in the recorded
patches rather than assumed. 124 is also exactly `17*7+5`, so the reference
video lost nothing to the grid snap.

**What is still open is the owner's other hypothesis**: that 124 frames is too
short for the density of dialogue in the source, so the model is asked to fit
a continuous monologue's speech into 5.167s. Nothing here tests it, and the
test is a length sweep on one prompt -- not a prompt comparison.

**The real finding is that reused-soundtrack speech degrades often** at this
canvas and length, across prompt registers. That is worth a designed
experiment and did not get one: the arms were confounded -- structure, length,
and whether the soundtrack is stated as an `<Audio 1>: fully_copy` retention
line or as prose in the body all moved together -- and four renders is not a
rate.

```
subject_definitions:
<Subject 1> is the character whose complete visual identity -- face, facial structure, eyes, skin tone, hair style and colour, body proportions, and overall appearance -- comes exclusively from <Picture 1>. Their body motion, posture, gestures, head movements, timing, and physical performance come from the original character in <Video 1>.
<Picture 1> supplies subject identity only. It does not supply lighting, exposure, colour grade, background, camera angle, pose, framing, or scene composition.
<Video 1> is the source video for the target video edit. It supplies the camera path, framing, background, environment, lighting, composition, action timing, and the original character's body motion. It does not supply the face or identity.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - facial structure, identity, hair, and appearance from <Picture 1> are retained.
<Video 1> (environment and motion): partially_preserved - the setting, lighting, and camera composition are retained, and the original character's actions are transferred to <Subject 1>.
```

**The negative clauses are not from the official guide.** Every relationship
there is stated as what a reference *provides*; nothing in it tells a
reference what it does not supply. They come from general prompting research,
where the reported failure is the model blending the two identities or
pulling the image's own lighting and background into the plate. Treat them as
an open question, not a rule -- that is why this arm sits beside
`h3_ref_video_edit` rather than replacing it.

Two practical notes. It wires **one** image, not two: the environment comes
from the plate, and a second reference would pay rows on every step to say
nothing. And a low-resolution reference, or a face far from the camera, is
the first thing to rule out when likeness fails -- the identity has to
survive being fitted into the reference budget before any wording matters.

Graph: `h3_ref_video_swap.json`.

### Continue from the end of a source video

`<Video 1>` is a starting state, not a thing to copy.

```
<Video 1> is the source video the target video continues from, beginning at its final frame.

retention_analysis:
<Video 1> (continuation source): partially_preserved - scene, lighting, and subject position continue from its final state.
<Subject 1> (appears in [Shot 1]): fully_preserved - face, hair, and clothing are retained from <Video 1>.

detailed_description:
[Shot 1] The shot begins exactly where <Video 1> ends, on the same framing and lighting, and carries the motion forward without a cut.
```

Graph: `h3_ref_video_continue.json`.

### Transfer motion onto a different subject

Motion does not ride on `<Video N>`. One subject, two assets, each named for
what it provides:

```
<Subject 1> is the person whose appearance comes from <Picture 1> and whose walking motion comes from <Video 1>.

retention_analysis:
<Subject 1> (appears in [Shot 1]): attribute_transfer - the gait and timing of <Video 1> are transferred to the person in <Picture 1>.
<Video 1> (motion source): attribute_transfer - only the gait and its timing are taken; the scene and the person are not.
```

Say explicitly that the video's own scene is not reused, or the model has two
competing environments.

Graph: `h3_ref_video_motion.json`.

### Follow camera movement, cuts and rhythm only

The weakest relationship, and the right one when images already supply the
subjects.

```
<Video 1> (cut and pacing structure): weak_reference - only the pacing of the camera move is followed.
```

Graph: `h3_ref_video_to_video.json` and the socket-combination arms.

### Reference a voice

Audio can carry a speaker's timbre and delivery, and the guide requires the
target speaker's **global speaker id**, not a new number:

```
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).

retention_analysis:
<Audio 1>: reference - only timbre and delivery are referenced, the signal is not copied.
```

`fully_copy` would ask for the source waveform itself, which is a different
request. Graph: `h3_ref_audio_voice.json`.

### Reference a music style

```
<Audio 1> is a standalone music reference whose tempo and instrumentation the target video's score follows.
<Audio 1>: reference - only tempo and instrumentation are referenced, the signal is not copied.
```

Graph: `h3_ref_image_audio.json`.

---

## The shipped arms

Nine reference graphs. The first five vary **which sockets are wired**; the
last four vary **what the prompt asks for**, holding the wiring roughly still.

| graph | images | video | soundtrack | audio | relationship |
|---|---|---|---|---|---|
| `h3_image_ref_plus_text_to_video` | 2 | | | | subjects from pictures |
| `h3_ref_video_only` | | yes | | | structure |
| `h3_ref_video_audio` | | yes | yes | | structure |
| `h3_ref_image_audio` | 2 | | | yes | music style |
| `h3_ref_video_to_video` | 2 | yes | yes | | structure |
| `h3_ref_image_video_audio` | 2 | yes | yes | yes | structure + music |
| `h3_ref_video_edit` | | yes | yes | | **edit** |
| `h3_ref_video_swap` | 1 | yes | yes | | **character swap** |
| `h3_ref_video_continue` | | yes | yes | | **continuation** |
| `h3_ref_video_motion` | 2 | yes | | | **motion transfer** |
| `h3_ref_audio_voice` | 2 | | | yes | **voice timbre** |

All load the `ref2va` checkpoint. One deliberate exception elsewhere:
`h3_probe_ref2v_turbo` runs `ref2va` with an `fl2v` distill LoRA — an
experiment, documented in its own note.

---

## Known limitations, collected

- **No mask.** Edits are prompt-driven whole-frame regeneration.
- **Native core has no fps input.** It assumes 24 twice. Shipped typed graphs
  own loader metadata and normalize, while retaining `force_rate=24` to keep
  migration media policy unchanged.
- **Reference video is never upscaled**, where the reference pipeline upscales.
- **Core never truncates reference audio**, where the reference pipeline
  truncates to the generated duration. Costs 80 rows per second of excess.
  **Locally handled in shipped graphs** by the typed compiler's derived cap;
  core's own behaviour is unchanged, so this stays a native limitation.
- **`ref_image_size='max'` does not upscale.** Neither mode does. It picks
  which ceiling sizes a reference down; `MiniMaxH3ReferenceFit` with
  `allow_upscale=True` is the only thing that raises a small one to it.
- **Reference video is truncated to the generated frame count**, so a short
  render cannot be conditioned on a long reference.
- **12-total and audio-never-alone are unenforced** by ComfyUI — and by
  sglang and DiffSynth-Studio. Only diffusers raises. See the reference-types
  section for where the limits come from.
- A silent clip's soundtrack output/input must be left unwired or the render dies.
- **Native core pairs a reference video's soundtrack by socket number, not by
  material.** ComfyUI matches `ref_video_audio_N` to `ref_video_N` on the name
  suffix (`comfy_extras/nodes_minimax_h3.py:313`), so a mis-numbered socket
  silently pairs the wrong track. sglang routes the video material itself into
  the audio encoder (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py:193-202`) and represents an
  absent soundtrack as a zero-length audio condition to keep block order, so
  the mistake is not expressible there. This repo's typed video record replaces
  suffix pairing with explicit ownership; native core remains unchanged.
- **No trim offset.** sglang takes a `start_time_seconds` on every material,
  video and audio alike. ComfyUI has no equivalent; trim upstream.
- **A mono reference on native core does not render — it raises.** Resolved 2026-08-21, run
  here on CPU against the real `pack_audio`, replacing the entry that had this
  as read-but-unverified, reproducible with `bench/check_mono_ref_audio.py`, a gate since 2026-08-21.
  ComfyUI's audio VAE preserves the input channel count
  (`comfy/ldm/minimax/audio_vae.py:427`) and `_encode_ref_audio` does not upmix
  (`comfy_extras/nodes_minimax_h3.py:71`), so a mono waveform yields
  `[1,32,1,T]` and `pack_audio` returns `T` rows. `PackedLayout` allocates
  `ref_audio_t * 2` slots for the block (`comfy/ldm/minimax/model.py:381-386`),
  so the masked assignment at `:659` fails with a shape mismatch. diffusers and
  DiffSynth-Studio both expand to stereo before encoding. Upmix or convert
  before the socket; the same path serves `MiniMaxH3AddGuide`, so an anchored
  mono soundtrack fails identically. This repo's typed compiler upmixes mono;
  that is local handling, not a core fix.
- At 1344x768 with images at `max`, one video reference does not fit on 24 GB
  past about 124 generated frames.
- **Confirmed identical, so nobody re-checks it**: the 2 fps subsample for the
  conditioner, including the pad of the index list to the temporal patch with
  the last value, and the merged-pair timestamp
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:706-718` against
  `comfy/text_encoders/minimax.py:170-179`).

**Read but not verified**, from the 2026-08-21 re-derivation and cheap to
close if it ever matters: what sglang does when a reference video is
**shorter** than its aligned target frame count. It truncates with
`ffmpeg -frames:v`, which simply returns fewer frames; whether the encoder then
re-aligns them was not traced. ComfyUI walks **down** to the nearest `17n+5`,
and so does DiffSynth-Studio.

## See also

- `docs/h3_ref2v_distillation.md` — why ref2v resists step distillation, and
  what to expect running it with an fl2v distill LoRA anyway.
- `docs/h3_resolutions.md` — the canvas rules the reference video inherits.
- `docs/checks.md` — what is guarded and what is not.
