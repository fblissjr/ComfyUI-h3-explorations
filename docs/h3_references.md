# MiniMax H3 references: every type, what it costs, how to prompt it

`ref2va` conditions on an ordered list of references. This is what each type
is, what ComfyUI actually does to it, what it costs, and how to write the
prompt so the model uses it the way you meant.

Sources: MiniMax's official prompt guide, general prompting research, the
diffusers reference pipeline, and ComfyUI's own code. Every number marked
**measured** was taken on this install against a live render; everything else
is read from source and says so.

Written 2026-08-13 against ComfyUI v0.33.0.

---

## The four reference types

`MiniMaxH3ReferenceToVideo` has four reference sockets. There is **no mask
socket and no fps input** on any of them.

| socket | type | max | what it is |
|---|---|---|---|
| `ref_images.ref_image_N` | IMAGE | 9 | a still, at high detail |
| `ref_videos.ref_video_N` | IMAGE (frame batch) | 3 | a clip, on the canvas rule |
| `ref_video_audios.ref_video_audio_N` | AUDIO | 3 | the soundtrack of the **same-numbered** video |
| `ref_audios.ref_audio_N` | AUDIO | 3 | a standalone audio asset |

Limits the reference pipeline enforces and **ComfyUI does not**: 12 references
total across all types, and an audio reference may never appear without at
least one image or video. Wire 15 and ComfyUI will accept it.

A reference video is an **IMAGE batch, not a VIDEO** — it arrives through a
frame loader, which is why the frame rate is your problem (below).

---

## What ComfyUI does to each one

### Image references

Scaled by `short_edge / min(w, h)` and rounded to 32. **ComfyUI clamps that
with `min(1.0, ...)` and the reference pipeline does not**, so a reference
smaller than 2048 on its short side reaches the DiT under-sized — and identity
fidelity is the whole job of a reference image. `MiniMaxH3ReferenceFit` exists
to close that gap; it needs the downstream `ref_image_size` on `max`, or the
stock node re-sizes from the video's pixel area instead and undoes it.

Refused outside 1:4..4:1. Image references are deliberately **exempt from the
768x1344 area cap** that binds the video, which is why one can legitimately
reach 7.5 megapixels when the video cannot exceed about one.

### Video references

1. Canvas from the reference's **own** aspect ratio via `adapt_canvas`.
2. **Never upscaled.** If the source has fewer pixels than that canvas,
   ComfyUI uses the source size rounded to 32. The reference pipeline puts it
   on the full canvas rule with no such clamp — the same divergence as image
   references, unclosed, because closing it costs about 5x what the image one
   does.
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

Resampled to the audio VAE's rate. **Not truncated.** The reference pipeline
cuts a soundtrack to the generated duration; ComfyUI encodes the whole
waveform, at 80 rows per second of excess. Trim it yourself.

---

## Frame rate: the one that bites silently

ComfyUI's node has **no fps input** and assumes 24 twice over — for the DiT's
temporal clock and for the `<T.T seconds>` labels the conditioner reads. The
reference pipeline resamples onto 24 from the rate the container reports.

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
| image at `max`, 1024x1024 source | 4,096 |
| image at `max`, 1280x720 source | 7,296 |
| video, 960x544 source, 124 frames | 18,870 |
| video, 960x544 source, 345 frames | **52,020** |

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

The failure is graceful and worth recognising: Sol-Attn's kernel OOMed and
fell back, then sage's OOMed and fell back, then ComfyUI's own SDPA OOMed.
Three clean degradations, each logged. There was simply no room.

Note the margin at the successful run — 21,938 of 24,564 MiB, about 2.6 GB
spare. Reference video is the most expensive input in the model.

**Budget by pixel area, not by count.** The same clip at 640x360 costs a third
of what it costs at 960x544.

---

## Labels: the tokenizer decides, not the prompt

References are emitted in a fixed order with a **separate 1-based counter per
type**:

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

`<Video N>` and `<Audio N>` are numbered independently, and an ordinary
reference video does not create an `<Audio N>` merely because the file has
sound. The soundtrack socket has to be wired.

**A silent clip must not have its audio socket wired.** VHS raises "failed to
extract audio" when its audio output is pulled on a video with no audio
stream, and the render dies at execution having validated cleanly.

---

## Prompt structure

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
| `h3_ref_video_continue` | | yes | yes | | **continuation** |
| `h3_ref_video_motion` | 2 | yes | | | **motion transfer** |
| `h3_ref_audio_voice` | 2 | | | yes | **voice timbre** |

All load the `ref2va` checkpoint. Two deliberate exceptions elsewhere:
`h3_image_ref_plus_text_to_video_ref_lora` runs `fl2va` plus an extracted
reference LoRA, and `h3_probe_ref2v_turbo` runs `ref2va` with an `fl2v` distill
LoRA — both experiments, both documented in their own notes.

---

## Known limitations, collected

- **No mask.** Edits are prompt-driven whole-frame regeneration.
- **No fps input.** 24 is assumed twice; use `force_rate=24`.
- **Reference video is never upscaled**, where the reference pipeline upscales.
- **Reference audio is never truncated**, where the reference pipeline
  truncates to the generated duration.
- **Reference video is truncated to the generated frame count**, so a short
  render cannot be conditioned on a long reference.
- **12-total and audio-never-alone are unenforced** by ComfyUI.
- A silent clip's audio socket must be left unwired or the render dies.
- At 1344x768 with images at `max`, one video reference does not fit on 24 GB
  past about 124 generated frames.

## See also

- `docs/h3_ref2v_distillation.md` — why ref2v resists step distillation, and
  what to expect running it with an fl2v distill LoRA anyway.
- `docs/h3_resolutions.md` — the canvas rules the reference video inherits.
- `docs/checks.md` — what is guarded and what is not.
