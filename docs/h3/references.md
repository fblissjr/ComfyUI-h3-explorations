# References: every type, what happens to it, what it costs

`ref2va` conditions on an ordered list of references: stills, videos (with or without their soundtrack) and audio
clips, each named in the prompt by a label. This page says what each type is, what ComfyUI does to it, what it costs
and where the known limits are, and links down:

- [`reference_sizing.md`](reference_sizing.md): how big a still is when it reaches the video model and the text
  encoder, the knobs that decide it, the vendor's image path stage by stage, and the encoder's pixel bounds.
- [`reference_prompts.md`](reference_prompts.md): a label is a bare ordinal, and how to write the relationships
  (edit, swap, continue, motion, structure, voice, music), with the shipped graph for each.
- [`conditioning.md`](conditioning.md): how every conditioning input is presented, encoded and positioned.
- [`comfyui_vendor_gaps.md`](comfyui_vendor_gaps.md): the catalogue of where core differs from the release, with
  status and enforcement.

The vendor-side authority compared against is **sglang's MiniMax H3 serving path** (`coderef/sglang`). diffusers is a
portability target (`h3_rules.reference_would_emit`), not a description of how the release is served; a sentence
here that says only "the reference pipeline" is a claim about diffusers. Numbers marked **measured** were taken on
this install against a live render; everything else is read from source and says so.

## The four reference types

Core's `MiniMaxH3ReferenceToVideo` has four parallel socket groups, with **no mask socket and no fps input**:

| socket | type | max | what it is |
|---|---|---|---|
| `ref_images.ref_image_N` | IMAGE | 9 | a still, at high detail |
| `ref_videos.ref_video_N` | IMAGE (frame batch) | 3 | a clip, on the canvas rule |
| `ref_video_audios.ref_video_audio_N` | AUDIO | 3 | the soundtrack of the **same-numbered** video |
| `ref_audios.ref_audio_N` | AUDIO | 3 | a standalone audio asset |

A reference video is an IMAGE batch, not a VIDEO: it arrives through a frame loader, which is why the frame rate is
your problem (below).

**This pack's typed surface.** Generated graphs wire `MiniMaxH3AppendRefImage`, `MiniMaxH3AppendRefVideo` and
`MiniMaxH3AppendRefAudio` into `MiniMaxH3ReferenceConditioning`. List position is explicit, the video record owns its
optional soundtrack and `VHS_VIDEOINFO`, and the compiler owns duration and channel normalization. The generator keeps
core's presentation order (images, then each sounded video, then standalone audio) so prompts keep their ordinals; a
hand-built chain can choose another order. Each append node shows the label it got once it has run.

**Budgets.** diffusers raises on more than 12 references in total
(`coderef/diffusers/src/diffusers/modular_pipelines/minimax_h3/before_encoder.py:410-413`) and on an audio reference
paired with nothing (`coderef/diffusers/src/diffusers/modular_pipelines/minimax_h3/before_encoder.py:414-418`).
ComfyUI, sglang and DiffSynth-Studio enforce neither: sglang's ref2va profile sets no count
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py:183-231`),
and DiffSynth raises only on an unknown kind and on a silent video passed as `video`
(`coderef/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py`). The limits come from MiniMax's Open
Platform API table (`coderef/MiniMax-H3/README.md:85`), which also carries per-clip 2-15 s and aggregate 15 s duration
limits nothing in ComfyUI enforces. VideoX-Fun's example `examples/minimax_h3/predict_ref2va.py` (PR #508, read
2026-08-27) states them per type as the released checkpoint's: at most 9 images, 3 videos, 3 audios, 12 in total, and
no audio alone. A comment in an example script, not an enforced control.

## Encoder-only references: leave the VAE unwired

A reference whose VAE is absent still reaches the text encoder exactly as before (same label, same vision block, and
the DiT still tags that text span as video modality,
`comfy/text_encoders/minimax.py::token_tags_from_embeds_info`, consumed in `comfy/ldm/minimax/model.py`), but gets no
reference-latent rows: no block is built and the sequence is laid out as for a text-only pass. Both VAE inputs are
optional on core's node (PR 16065, core `1aec3a13`) and on `MiniMaxH3ReferenceConditioning`;
`bench/check_reference_runtime.py::encoder_only_references_skip_the_dit_rows` holds ours to core's three cells. For
one still, `MiniMaxH3AppendRefImage.use_vae` (on by default) does the same per reference.

| kind | without the video VAE | without the audio VAE |
|---|---|---|
| still | Qwen view only, no `ref_img` rows | unchanged |
| video | Qwen's 2 fps samples only, no rows; its soundtrack label stays and its audio latent is never built | label stays, silent `video` block |
| standalone audio | unchanged | its `<Audio j>` label and nothing else |

The encoder is never handed audio, so an audio reference with no audio VAE conditions on a bare ordinal; core's gate
order is the same, a sounded video needing the video VAE before its soundtrack is considered.

**Why anyone would want it.** Reference rows are the expensive input, attended every step; the encoder pathway costs
tokens in the text segment only. One render with its only still encoder-only looked the same to the owner and sampled
faster (`bench/results/2026-10-03_encoder_only_reference.md`, with what one unblinded clip does not establish); no
shipped graph turns it off. How much of the result the rows carry is unmeasured: `bench/ref_pathway_arms.json` is the
five-arm ablation built for it (the `h3_probe_ref_pathway_*` family; `ref_latents=False` and `native_ref=True` are the
builder's knobs). `bench/preflight_graph.py` reports an encoder-only conditioner as zero DiT reference rows. Not the
release's path: the vendor pipelines always build the latents.

## Encoding references apart from the prompt

`MiniMaxH3ReferenceConditioning` encodes the references and the prompt as one sequence, so a prompt edit pays for the
references again, nearly all of it the text encoder reading the stills
(`data/results/2026-10-03_prompt_edit_conditioning_cost.json`). Core puts every reference ahead of the prompt and the
encoder is causal, so the references' encoding cannot depend on the prompt. Two experimental nodes split the work
there (`reference_encode.py`):

- **MiniMax H3 Encode References** takes the encoder, the VAEs and the chain and encodes the references alone, each
  continued on the ones before it. It does not take the prompt, so a prompt edit leaves it in core's node cache. Its
  preview gives each reference's token count and time.
- **MiniMax H3 Prompt On References** takes that output and the prompt and encodes only the prompt's tokens on top,
  with the encoder the references were encoded with.

Nothing is kept between runs: the output holds the rows on the host while wired, never on the card and never on disk
(the owner's rule that nothing an earlier run made is reused by a key, `docs/state/decisions.md`, 2026-10-10). The
result is not bit-identical to the one-node path; it differs by far less than the int8 encoder differs from bf16
(`data/results/2026-10-03_reference_split.json`; the mechanism in `2026-10-03_encoder_prefix_reuse.json`), which the
owner accepted on 2026-10-03. No shipped graph wires these nodes. The two RefMod packs under `coderef/` pre-encode
differently, encoding the prompt without the image in context, which is a different conditioning (the same record's
`prompt_without_image` case).

## What ComfyUI does to each one

### Image references

Scaled and rounded to 32, and exempt from the video's area cap, which is why one can reach several megapixels when
the video cannot exceed about one. Refused outside 1:4..4:1. A still is read twice: the video VAE's copy becomes DiT
reference rows attended every step (`size_policy` sizes it), the text encoder's copy becomes vision tokens ahead of
the prompt (`qwen_view` sizes it). The defaults are the append node's `define_schema`: the video model's copy at
vendor parity (a 2048 short edge, upscaling allowed), the text encoder's copy `separate` at
`h3_rules.REF_QWEN_SHORT_EDGE`, the owner's choice of 2026-10-03 and a departure from the vendor. Core's own node never
upscales. All of it: [`reference_sizing.md`](reference_sizing.md).

### Video references

1. Canvas from the reference's **own** aspect ratio via `adapt_canvas`.
2. **Never upscaled.** If the source has fewer pixels than that canvas, core uses the source size rounded to 32
   (`comfy_extras/nodes_minimax_h3.py:316-320`). All three vendor implementations put it on the canvas rule with no
   such clamp: sglang through the same shape function it uses for the target
   (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/prequeue.py:292-302`),
   DiffSynth-Studio by `min(768/min(w,h), sqrt(max_pixels/(w*h)))`, diffusers through its canvas resolver. The typed
   conditioner's `video_policy=release` applies the vendor's view and its coupled Qwen stage
   ([`reference_sizing.md`](reference_sizing.md), "Video and audio"); generated graphs use `comfy`, core's behavior.
3. Truncated to the **generated** frame count, then snapped **down** to the `17n+5` grid. Fewer than 5 frames raises.
   So a short render can only ever be conditioned on a short reference.
4. VAE-encoded whole. Those rows ride **every sampling step**.
5. Subsampled to **2 fps** for the text encoder, each merged frame pair labelled `<T.T seconds>`; identical to sglang,
   index pad and pair timestamp included.

### Audio references

Resampled to the audio VAE's rate. **Core does not truncate**: it encodes the whole waveform, at 80 rows per second
of excess. sglang cuts every reference soundtrack to the generated duration
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/audio_encoding.py:147-151`)
and diffusers does too (`coderef/diffusers/src/diffusers/modular_pipelines/minimax_h3/before_encoder.py:346`);
DiffSynth-Studio does not. Core is inconsistent with itself: a keyframe's audio is truncated to the remaining track
and raises past the end (`comfy_extras/nodes_minimax_h3.py:226-230`).

**Rows are not the only consequence.** A video reference advances the packed rotary cursor by
`max(ref_audio_t, sum of video spans)`, so a soundtrack longer than its own visual reference expands the reference
span and pushes the target streams down the timeline (both independent reviews reach it:
Codex's of 2026-08-21, section 5.3, and the cursor formula in
`internal/gemini/minimax_h3_comfyui_end_to_end_trace_and_gap_analysis.md`). Trimming is a correctness change, not
only a saving. `length / 24` is the exact cap: `_ref_t_span` compares the two in the same units
(`comfy/ldm/minimax/model.py:105-113`), and a soundtrack trimmed to the render lands on its own video's span. Left
untrimmed, a 19.541 s source against a 124-frame render claims 3.8x the timeline it should (computed 2026-08-22 with
`temporal_shape` and `_video_t_spans` imported).

**Every shipped graph is capped at the typed boundary**: the compiler derives `frame_count / 24`, slices an owned
soundtrack and standalone audio alike, duplicates mono to stereo, and refuses more than two channels. A native socket
graph is still exposed; preflight reports its trim and channel state. On the video-soundtrack path VHS also caps:
both its upload loaders ask ffmpeg for `frame_load_cap * (1 / force_rate)` seconds of audio
(`custom_nodes/ComfyUI-VideoHelperSuite/videohelpersuite/load_video_nodes.py::load_video` into
`custom_nodes/ComfyUI-VideoHelperSuite/videohelpersuite/utils.py::get_audio`, paths relative to the ComfyUI root;
measured 2026-08-22: cap 0 yields the whole track, cap 124 yields 5.167 s). The compiler cap stays authoritative: it
survives a loader cap set back to 0 and is the only mechanism for standalone `ref_audio_*`.

## Frame rate: the one that bites silently

Core has **no fps input** and assumes 24 twice, for the DiT's clock and for the `<T.T seconds>` labels
(`FPS = 24` at `comfy_extras/nodes_minimax_h3.py:30`). sglang's constant is the same
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py:29`)
but it is applied as an ffmpeg `fps=` filter in the decode pass
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:397`),
so a 30 fps source really is converted to 24 before anything sees it, and fps never enters its API: the caller asks
for `duration_seconds`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py:291-295`).
The decoded array is shared by Qwen and the VAE by construction
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:380-383`).

**Measured**, three 6.00-second clips differing only in frame rate:

| source | `force_rate` | H3 reads it as | error | last label |
|---|---|---|---|---|
| 24 fps | 0 or 24 | 5.875s | 0.0% | `<5.2 seconds>` |
| 25 fps | 0 | 5.875s | **+4.2%** | `<5.2 seconds>` |
| 30 fps | 0 | **7.292s** | **+25.0%** | `<7.0 seconds>` |
| any | 24 | correct | 0.0% | correct |

**Set `force_rate=24` on the loader** for a reference whose exact frames do not matter. A 24 fps source is unaffected
either way, which is why testing on one proves nothing; `bench/check_ref_prompt_labels.py` fails the build if a loader
feeding a reference drops off 24. The typed `MiniMaxH3AppendRefVideo` also records the loader's `loaded_fps` and the
conditioner normalizes to 24.

**But `force_rate=24` gets the labels right by dropping or repeating frames.** On a 25 fps source it drops one frame
a second, measured on 2026-10-08 as a doubled camera step on those frames and seen by the owner as jitter. A masked
source, or any video the result must stay in step with, goes to the loader as an every-frame 24 fps copy (the same
frames re-stamped, nothing resampled) with `force_rate` 0, and the delivery is re-stamped to the source's own rate
with its own audio (`bench/assemble_delivery.py`).

## What references cost

Reference rows are attended at **every sampling step**, exactly as video rows are. Measured against a live render at
1344x768, the DiT reference block:

| reference | rows |
|---|---|
| audio, per second | 80 |
| image at `match` | ~1,008 |
| image, 1024x1024 source, upscaled to 2048 | 4,096 |
| image, 1280x720 source, upscaled to 2048 | 7,296 |
| video, 960x544 source, 124 frames | 18,870 |
| video, 960x544 source, 345 frames | **52,020** |

**A still costs in two places.** The text encoder reads it too, as vision tokens in the text segment. Measured
2026-08-13 on one graph with both copies shared, upscaling one 1024x1024 still to 2048 added 3,072 rows to the
reference segment and 3,072 to the text segment. A ladder of 1/2/3/4/6 references at the sizing of 2026-08-15 (a
peer session's measurement, not re-derived) gave text rows of 4,171 / 19,778 / 28,002 against reference rows of
4,096 / 19,648 / 27,840 at 1, 4 and 6 references, the prompt under 200 tokens of the text column. Nine references at
that sizing is roughly 94,000 rows and OOMs on 24 GB. With `qwen_view=separate`, the default, the text copy is
smaller than the video model's copy and the two costs part. **Price a graph's two copies with
`bench/preflight_graph.py`**, which reports each separately and says when one was clamped and by whose bounds, or read
the conditioner's on-node preview; not with a rule of thumb from this page.

**A reference video costs in two places too**: the DiT block above, and its 2 fps frame pairs as vision blocks in the
text segment, roughly 519 tokens a block. Going from 124 to 345 reference frames grew the text segment by 4,667 tokens
on top of 33,150 extra reference rows.

### It does not all fit

**Measured on a 24 GB card**, images at `max` plus one video plus its soundtrack:

| generated | sequence | result |
|---|---|---|
| 124 frames | 78,019 | **success**, 740s, peak 21,938 MiB |
| 345 frames | 182,092 | **OOM** at step 4 of 16, 21.05 GiB allocated |

Treat this as the shape of the cliff, not as clearance: the video-reference arms render at `h3_config.LONG_LENGTH`,
and a safe length depends on reference count, kind, duration, canvas and upscale, so no constant is right twice. Read
preflight before any reference render. The failure was graceful: Sol-Attn's kernel, then sage's, then ComfyUI's SDPA
each ran out and fell back, logged. Reference video is the most expensive input the model takes.

**Budget by pixel area, not by count.** The same clip at 640x360 costs a third of what it costs at 960x544.

## Labels: the tokenizer decides, not the prompt

Core emits native socket references in a fixed order with a **separate 1-based counter per type**:

1. images, as `<Picture i>`
2. then each video: its paired soundtrack's `<Audio j>` **immediately before** its `<Video k>`
3. then standalone audio, continuing the `<Audio j>` count

So one video with sound plus one standalone clip reads `<Audio 1>`, `<Video 1>`, `<Audio 2>`. Getting it wrong is
silent: the render succeeds and quietly ignores an instruction about something that is not there.
`bench/check_ref_prompt_labels.py` asserts every shipped graph's prompt names exactly what its graph wires, in this
numbering, and also catches a wired reference the prompt never mentions, which still costs its rows every step.

The typed surface replaces fixed grouping with list position and suffix pairing with ownership; the counters and the
sounded-video rule are unchanged. An ordinary reference video does not create an `<Audio N>` because its file has
sound: the soundtrack has to be wired. **A silent clip must not have its soundtrack pulled**: VHS raises "failed to
extract audio" at execution, after the graph validated cleanly.

What the labels mean and how to write the prompt around them: [`reference_prompts.md`](reference_prompts.md).

## Known limitations, collected

Core's behavior; where this pack handles one locally it says so.

- **No mask.** Edits are prompt-driven whole-frame regeneration. (The masked lane is not a reference route:
  [`../masked_v2v/overview.md`](../masked_v2v/overview.md).)
- **No fps input**; core assumes 24. Typed graphs normalize from loader metadata and keep `force_rate=24`.
- **Reference video is never upscaled**, where the vendor puts it on the canvas rule.
- **Reference audio is never truncated**, where sglang and diffusers truncate. Capped locally by the typed compiler.
- **Core never upscales a still** in either `ref_image_size` mode; on the typed path `allow_upscale` does
  ([`reference_sizing.md`](reference_sizing.md)).
- **Reference video is truncated to the generated frame count.**
- **The 12-total and audio-never-alone budgets are unenforced** by ComfyUI, sglang and DiffSynth; only diffusers
  raises.
- **A silent clip's soundtrack must be left unwired**, or the render dies.
- **Core pairs a video's soundtrack by socket number**, matching `ref_video_audio_N` to `ref_video_N` on the suffix
  (`comfy_extras/nodes_minimax_h3.py:313`), so a mis-numbered socket silently pairs the wrong track. sglang routes the
  video material itself into the audio encoder
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py::MINIMAX_H3_TASK_PROFILES`,
  the `video` rule's `audio_tokenizer_encode`) and keeps an absent soundtrack as a zero-length condition
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/audio_encoding.py::MiniMaxH3AudioEncodingStage`).
  The typed record uses ownership instead.
- **No trim offset.** sglang takes a `start_time_seconds` on every material; ComfyUI has none, so trim upstream.
- **Mono works; more than two channels is silently cut to the first two.** Core's `comfy.sd.VAE.encode` runs
  `vae_encode_crop_pixels` first, and the H3 audio VAE declares `output_channels = 2` and
  `pad_channel_value = "replicate"` (`comfy/sd.py:1030-1035`), so mono is duplicated to stereo (sglang does the same
  with `-ac 2`) and a 5.1 track becomes its first two channels with no warning. Measured against the real audio VAE
  ([`../../data/results/2026-08-29_ref_audio_channels.json`](../../data/results/2026-08-29_ref_audio_channels.json),
  [`../../bench/audit_ref_audio_channels.py`](../../bench/audit_ref_audio_channels.py)). The typed
  `reference_conditioning._prepare_audio` refuses more than two channels.
- At 1344x768 with images at `max`, one video reference does not fit on 24 GB past about 124 generated frames.
- **Read, not verified**: what sglang does with a reference video shorter than its aligned target. It truncates with
  `ffmpeg -frames:v`; whether the encoder re-aligns was not traced. ComfyUI and DiffSynth walk down to the nearest
  `17n+5`.

## See also

- [`../distills/ref2v_distillation.md`](../distills/ref2v_distillation.md): why ref2v resists step distillation.
- [`geometry.md`](geometry.md): the canvas rule a reference video inherits.
- [`../method/checks.md`](../method/checks.md): what is guarded and what is not.
