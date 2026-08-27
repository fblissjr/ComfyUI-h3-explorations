# How every H3 conditioning input is handled, end to end

Last verified against the installed ComfyUI and this repo's nodes on 2026-08-25.

Five things can condition an H3 render: a **keyframe** (first frame, last frame,
or any frame via `MiniMaxH3AddGuide`), a **reference still**, a **reference
video**, a **reference video's own soundtrack**, and a **standalone reference
audio**. They are not variations of one mechanism. They differ at the text
encoder, at the VAE, in where they land on the DiT's rotary clock, and in
whether they are denoised at all.

This file is the map. Where a number or rule is owned elsewhere, it links rather
than restating: [`h3_references.md`](h3_references.md) owns reference types,
sizing knobs and measured costs;
[`h3_geometry_and_nodes.md`](h3_geometry_and_nodes.md) owns the frame grid and
token maths; [`h3_awq_encoder.md`](h3_awq_encoder.md) owns the AWQ adapter's
preprocessing.

## The one-screen answer

| | keyframe | reference still | reference video | video soundtrack | standalone audio |
|---|---|---|---|---|---|
| Qwen text | `<Picture i>:` | `<Picture i>:` | `<Video k>:` + `<T.T seconds>` per pair | — | `<Audio j>:` |
| Qwen tensor | vision block | vision block | one block per frame pair | none | none |
| geometry | **target canvas**, forced | its own, policy-dependent | its own, policy-dependent | n/a | n/a |
| VAE | video VAE | video VAE | video VAE, whole clip | audio VAE | audio VAE |
| rotary time | **on the target timeline** at `cursor + 5/3 × frame_index` | own slot, `+1.0` | own slot, `+max(audio, video span)` | shares its video's origin | own slot, `+ref_audio_t` |
| spatial grid | **the target's** | its own | its own | n/a | n/a |
| denoised? | **no** — fixed every step | no | no | no | no |

Two facts drive most of the table. A keyframe **is** a frame of the output, so it
sits at the output's time coordinate on the output's grid. Everything else sits
*before* the output timeline with its own grid. And no conditioning row is ever
denoised — only the target rows are.

**"Policy-dependent" is not a hedge.** A keyframe's geometry is forced by the
packing; a reference's is not, so it is decided by settings that vary per graph
and per reference, then capped again by the encoder. The 2048 short edge is the
*vendor serving convention*, and installed ComfyUI caps at it without ever
upscaling to it, so a reference reaches that size here only when
`MiniMaxH3AppendRefImage(allow_upscale=True)` is wired, which some shipped
graphs do and most do not. Reference video is sized by its own separate policy again. Never
quote a single number for reference geometry; read
[`h3_references.md`](h3_references.md) for the knobs and
[`2026-08-24_serving_geometry_composes.md`](research/qwen3-vl-special-tokens-post-training/canonical/2026-08-24_serving_geometry_composes.md)
for how upstream sizing and the encoder cap compose.

For Qwen's reference-video view, native stock ComfyUI and this repo's shipped
graphs are also different paths. Stock applies shared bounds independently to
each two-frame block. Shipped graphs other than the `release` probe arm select
`video_policy=encoder`, which keeps the no-upscale VAE view but applies the
loaded encoder's clip-wide, duration-aware Qwen stage, read off the CLIP's
stamped contract (a CLIP that declares nothing resolves to native, logged);
`release` additionally applies vendor VAE sizing and
`comfy` remains the native control. The source/length boundary where the stock
and clip-wide grids actually separate is measured—not universal—and is owned
by [`comfyui_vendor_gaps.md`](comfyui_vendor_gaps.md) section 2.

## 1. What Qwen3-VL sees

**SOURCE:** `comfy/text_encoders/minimax.py::MiniMaxH3Tokenizer.tokenize_with_weights`.
The presentation is raw, never chat-templated. Reference items are emitted in
request order with an independent one-based counter per type, then the prompt
text follows everything.

```
t2va    <prompt>
fl2va   "<Picture 1>: " <vision> ["<Picture 2>: " <vision>] <prompt>
ref2va  per item in request order:
          image  ->  "<Picture i>: " <vision>
          audio  ->  "<Audio j>: "                    (no tensor)
          video  ->  "<Video k>: " then per frame pair:
                       "<T.T seconds>" <vision(2 frames)>
        then <prompt>
```

Three consequences worth holding onto:

**A keyframe and a reference still carry no explicit role field here.**
`MiniMaxH3ImageToVideo` calls `clip.tokenize(prompt, images=images)` and never
passes the resolved frame index. Both emit `"<Picture i>: "` plus a vision
block. Their pixels, geometry, surrounding prose and position may differ, but
all else equal the encoder receives no structural
keyframe-versus-reference-still indicator.

**SOURCE: prose is the only explicit encoder-side carrier of a keyframe's
target time.** The resolved frame index is never passed to the tokenizer, so
phrases like "at 0.00 seconds into the target video, `<Picture 1>` is fully
referenced" are the only channel by which that information *could* reach the
encoder. Reference *video* is different: its timing is structural, carried by
the `<T.T seconds>` markers.

**UNKNOWN: whether the encoder uses that prose.** "Only channel available" is
not "the model reads it." Nothing here measures whether omitting the timing
phrase changes the conditioning, or whether the encoder recovers a keyframe's
role some other way. The DiT's anchor is unaffected either way — that part is
structural.

**SOURCE: the reference labels are ordinary BPE, not special tokens.**
`<Picture 1>`, `<Audio 1>`, `<Video 1>` go through the normal tokenizer. They
carry adaLN token tag 1, the same tag as `<d>` and `</d>`; only the vision spans
carry tag 0. So a dialogue marker and a reference label sit adjacent in one text
stream, sharing a tag.

**INFERENCE, and no further.** That adjacency makes an interaction *possible*
and is a reason to measure one before any post-training moves the seven marker
embeddings. It does not establish that the labels are what carries
prompt-to-vision-block binding, nor that moving marker embeddings would change
that binding. Both remain UNKNOWN; neither has been measured here or anywhere in
this repo.

Audio never reaches Qwen as a tensor at all. It contributes a text label and
nothing else.

## 1b. Qwen3-VL as a pipeline stage: inputs, transformations, outputs

Section 1 is what the encoder is shown. This is what happens to it from there
to the DiT, for the Ref2VA route, as the installed code does it. SOURCE unless
marked; the two inferences are marked because the record refuses to promote
them.

```
INPUTS (per request, in the order the references were listed)
  reference still i ── stage 1 sizing, per reference (match | max | fit) ──────┐
  reference video k ── 24 fps normalise, 17n+5 snap, canvas rule, 2 fps sample ┤
  reference audio j ── nothing: a text label only ────────────────────────────┤
  prompt text (the H3 prompt) ─────────────────────────────────────────────────┤
                                                                               ▼
PRESENTATION (comfy/text_encoders/minimax.py; raw, never chat-templated)
  "<Picture 1>: " <vision>  "<Audio 1>: "  "<Video 1>: " "<0.2 seconds>" <2 frames> ...  prompt
  labels are ordinary BPE, tag 1; every vision span with its start/end sentinels, tag 0
                                                                               ▼
STAGE 2, per vision block: the selected Qwen processor (release, v1 snapshot, or comfy)
  resize to its pixel bounds; 16-pixel patches; temporal patch 2 (a still repeats its
  frame); 2x2 merge -> grid_thw and a patch tensor; timestamps for video blocks
                                                                               ▼
QWEN3-VL
  vision tower: 27 blocks, attention within one image, BF16 weights, FP32 compute
     -> merged visual embeddings, plus three DeepStack taps (blocks 8, 16, 24)
  token embedding of the text ids; image-pad positions replaced by the visual embeddings
  M-RoPE positions: 3-D over vision spans, 1-D over text
  decoder layers 0..49 of 64; DeepStack features added after layers 0, 1 and 2;
  text tokens and image tokens attend to each other, causally, over one sequence
                                                                               ▼
OUTPUT TAP: the raw residual after layer 49; no final norm, no LM head
  one 5120-wide vector per token of the whole presentation, images included,
  travelling with the per-token tag the encoder never saw
                                                                               ▼
DiT ENTRY (comfy/ldm/minimax/model.py)
  condition_proj 5120 -> 5376, a two-layer token refiner; the rows become the packed
  sequence's text prefix, positioned 1-D, tag-selected AdaLN, attended by every target
  row at every denoising step, never denoised
```

Beside it, the VAE branch takes the **stage-1** image, never the stage-2 view,
encodes it with the video VAE, and packs those latents as reference rows with
their own rotary slot and their own grid (sections 2 and 3). The two branches
meet only inside the DiT's attention.

Read as a pipeline:

- **Sources.** Pixels from the reference files after stage-1 sizing; text from
  the H3 prompt; timestamps derived from the 2 fps sample indices; nothing
  from audio.
- **Transformations inside Qwen.** Patchify and merge, which is geometry;
  the vision tower and DeepStack, which are learned; then the decoder layers,
  in which prompt tokens and image tokens condition each other. The only place
  the words "`<Picture 1>`" and the pixels of picture 1 meet is that attention.
  **INFERENCE:** that this is where prompt-to-picture binding happens; the
  record keeps it unpromoted (section 1).
- **What passes through untouched.** The token tags. Qwen never sees them;
  they ride beside the sequence to the DiT.
- **What it emits, and to where.** One vector per token from layer 49, the tap
  the DiT's projection and refiner were trained against; layers 50 to 63 are
  never used. There is no separate image embedding, no pooled vector, and no
  spatial grid carried through: an image is a run of conditioning rows in text
  order.
- **Where the calibration geometry bites.** Stage 2 is the only place the v1
  artifact and a v2 candidate differ for a still (the v1 snapshot's pixel cap
  against release bounds), and the layer-49 vectors are the object AWQ's
  per-channel scales are fitted to. The geometry Qwen sees during calibration
  has to be the geometry it sees at serving time; the VAE branch does not enter
  that choice.
- **The two branches need not share a geometry.** Each is exact within itself
  (Qwen's 32-pixel grid; the VAE's 32-pixel multiple and 17n+5 frames), and
  nothing indexes a Qwen token against a latent patch. The deployed v1 path
  has run "VAE fine, Qwen coarse" since it shipped, and video is "Qwen at
  2 fps pairs, VAE at 24 fps" by design. A "Qwen at 2048, VAE at source"
  serving mode therefore breaks no contract; what it changes is a quality
  question. It exists since 2026-08-25 as
  `MiniMaxH3AppendRefImage.qwen_short_edge`
  ([`h3_references.md`](h3_references.md), "A third knob"): the VAE keeps
  the stage-1 view, the encoder is shown the source at an N short edge, and
  the loaded encoder's bounds still apply to that view afterwards. **INFERENCE:** appearance from the VAE rows, meaning and binding
  from the Qwen rows, in some unmeasured proportion; the blind comparison
  that would measure it has not been run
  ([`eval_comparison.md`](eval_comparison.md) section 3).

## 2. What the VAE does

Two VAEs, and which one runs depends only on the modality.

**Video VAE** encodes keyframes, reference stills, and reference video. A still
is encoded as a one-frame clip; a reference video is encoded **whole**, not
sampled. **SOURCE:** `vae.encode(kf.pop("image"))` for keyframes,
`vae.encode(resized)` per still, and the video branch in
`reference_conditioning.py::_compile_reference_records`.

**Audio VAE** encodes a video's soundtrack and standalone audio alike, through
`_encode_ref_audio`, which returns the latent and its length `ref_audio_t`.

The important asymmetry: **the video VAE and Qwen see different pixels of the
same reference.** The VAE takes whatever the reference nodes produced; Qwen
takes that *through* the encoder's own image processor. Under the current AWQ
artifact the second stage caps at 200,704--301,056 px, so a reference can reach
the DiT at full requested geometry and the conditioner at a fraction of it. That
composition is owned by
[`2026-08-24_serving_geometry_composes.md`](research/qwen3-vl-special-tokens-post-training/canonical/2026-08-24_serving_geometry_composes.md).

Reference video is also **truncated to the generated frame count** and snapped
*down* to the `17n+5` grid before encoding, so a short render can only ever be
conditioned on a short reference.

## 3. How it is positioned

**SOURCE:** `comfy/ldm/minimax/model.py::PackedLayout`. One packed sequence holds
text, conditioning and targets. Rows are laid out in this order:

```
text │ keyframe conds │ reference blocks │ target audio │ target video
```

but the **rotary time** each occupies is the part that matters:

```
text        0 ────────────────────► text_len
references  text_len ──────────────► cursor      (each advances the clock)
              still            +1.0
              audio            +ref_audio_t
              video            +max(ref_audio_t, Σ video spans)
keyframes   cursor + 5/3 × resolved_frame_index   ◄── same origin as the target
targets     cursor ─────────────────►
```

`cursor` is `text_len` plus every reference's span. Keyframes and the generated
video **share that origin**, so a keyframe at frame 0 occupies exactly the
target's frame-0 coordinate. That is what "a keyframe is a frame of the output"
means mechanically.

`FRAME_RESCALE = 5/3` converts a pixel-frame index into rotary time units.
`FRAME_PER_TOKEN = (1, 4, 4, 4, 4)` is how many pixel frames each latent token
covers, cycling — so a video reference's span is
`Σ 5/3 × FRAME_PER_TOKEN[k mod 5]`, not simply proportional to its length.

Spatial grids differ too: a keyframe uses the **target's** `frame` grid, shared
with the generated rows. Every reference builds its own from its own
`latent_h`/`latent_w`. That is the structural reason a reference may carry more
spatial detail than the video will ever be generated at, and a keyframe may not.

A video reference's soundtrack rows pack **immediately before** its video rows,
both sharing that reference's cursor origin.

## 4. How it conditions the DiT

**SOURCE:** `comfy/model_base.py`. Everything H3-specific rides in one
`minimax_payload` dict, deliberately kept out of the model's dtype cast so fp32
conditioning latents and long token tags are not flattened to bf16. Keyframes
arrive under `minimax_keyframes`, references under `minimax_refs`; both
contribute to flat `cond_video_latents` and `cond_audio_latents` lists, with
**keyframe latents first**.

**Conditioning rows are never denoised.** `PackedLayout` builds boolean update
masks: every keyframe and reference row gets `torch.zeros` — excluded from the
update — while target audio and target video get `torch.ones`. The conditioning
rows are re-injected at each sampling step instead, pinned at fixed timesteps
(`VISUAL_COND_TIMESTEP = 0.999`, `AUDIO_COND_TIMESTEP = 1.0`).

This is why reference cost is not a one-off: **reference rows ride every step**,
so their token count affects every denoising step. Its share of total render
time must be measured for the workload in question.

## 5. What this means in practice

**Sizing is per reference, and should stay that way.** Each record carries its
own policy, geometry, latent grid and rotary slot, and
`_compile_reference_records` applies them independently. Mixed geometry does not
break anything. An identity reference can hold real resolution while a
background or style reference stays cheap. More rows certainly increase packed
sequence length and attention cost at every step; whether they also increase a
reference's effective conditioning influence is unmeasured.

**A keyframe's spatial size cannot be reduced with the per-reference still
policy.** It is pinned to the canvas by the packing. Canvas size and the number
of guides remain separate cost levers.

**Contradictory sizing settings are no longer expressible.** `size_policy`,
`short_edge` and `allow_upscale` came onto one node on 2026-08-24, which made a
`match` policy carrying a non-default `short_edge` *checkable* — it warned. On
2026-08-27 `size_policy` became a `DynamicCombo` and the two knobs moved inside
the `max` branch, so selecting `match` removes them from the node entirely. The
warning went with them: a state that cannot be reached does not need reporting,
and a warning that only fires after a render is queued was the weaker fix. See
[`h3_references.md`](h3_references.md) for the two knobs and how they are
confused for each other.

**Both branches can coexist.** The layout computes the target origin *after*
reserving the references' span, then overlays keyframes on the target timeline.
A request carrying both is structurally supported even though the two come from
different nodes.

## Bounds

- Read from installed code and this repo's nodes. Nothing here was executed
  against a running model, and no claim is a numerical or perceptual result.
- The cross-implementation corroboration for the packing rule lives in
  [`2026-08-24_keyframe_vs_reference_positioning.md`](research/qwen3-vl-special-tokens-post-training/canonical/2026-08-24_keyframe_vs_reference_positioning.md);
  this file does not restate its evidence.
- Nothing here establishes what any of it is *worth*. Whether a larger reference
  conditions better, whether keyframe prose matters to output quality, and
  whether per-reference sizing variation is visible are all unmeasured.
