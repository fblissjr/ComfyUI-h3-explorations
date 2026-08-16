# MiniMax H3 references: every type, what it costs, how to prompt it

`ref2va` conditions on an ordered list of references. This is what each type
is, what ComfyUI actually does to it, what it costs, and how to write the
prompt so the model uses it the way you meant.

Sources: MiniMax's official prompt guide, general prompting research, the
diffusers reference pipeline, and ComfyUI's own code. Every number marked
**measured** was taken on this install against a live render; everything else
is read from source and says so.

Written 2026-08-13 against ComfyUI v0.33.0. Reference-image sizing
re-read from source and corrected 2026-08-16.

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

Scaled and rounded to 32. **ComfyUI clamps the scale with `min(1.0, ...)` and
the reference pipeline does not**, so a reference smaller than 2048 on its
short side reaches the DiT under-sized — and identity fidelity is the whole job
of a reference image. `MiniMaxH3ReferenceFit` exists to close that gap.

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
| lives on | `MiniMaxH3ReferenceToVideo` | `MiniMaxH3ReferenceFit` |
| applies to | **all image references at once** | **one reference** |
| touches video refs | no | no |
| touches audio refs | no | no |

`ref_image_size` is read in exactly one loop, the `ref_images` one. Video
references never see it — they take `adapt_canvas` from their own aspect ratio
(`:315-319`) — and audio has no spatial sizing at all.

`MiniMaxH3ReferenceFit` takes **one image per node** and warns if you feed it a
batch (`reference_fit.py:165-168`), so upscaling is decided per reference. Five
references means five nodes and five independent decisions.

### The four combinations

| ReferenceFit | conditioning node | what happens |
|---|---|---|
| `allow_upscale=True` | `max` | 2048 survives. **The working combination.** |
| `allow_upscale=True` | `match` | upscale **undone** — core re-shrinks to the generation's pixel area, and you paid two lanczos resamples for nothing |
| `allow_upscale=False` | `max` | both no-ops for anything under 2048 |
| `allow_upscale=False` | `match` | reference sized to the generation's pixel area |

Row two is why the node inspects its consumer at run time
(`reference_fit.py:199-206`) and warns. Row three is the state **10 of the 18
shipped graphs that wire ReferenceFit are in**, deliberately — `REF_VIDEO_BUDGET`
holds `allow_upscale=False` to fit 24 GB. As of 2026-08-16 the node says "NO
CHANGE" in the log when it lands there, because its presence in a graph
otherwise reads as "the references were fitted".

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

### Should video and audio have fit nodes too?

Both have the same kind of divergence from the reference pipeline. The answers
go opposite ways.

**Audio: yes, and it already exists — it is a trim, not a fit.** ComfyUI
encodes the whole waveform (`_encode_ref_audio`, no truncation on either the
soundtrack or the standalone path) where the reference pipeline cuts a
soundtrack to the generated duration. So here ComfyUI does **more** than the
reference, at 80 rows per second of excess, and closing the gap *saves* rows
while moving toward the reference's behaviour. Core already ships
`TrimAudioDuration` (`comfy_extras/nodes_audio.py:430`), so this is a wiring
fix and a Preflight warning, not a new node.

**Video: no, not now.** The divergence is real and the same shape as the image
one — never upscaled, where the reference puts the clip on the full canvas rule
— but building it would make **the most expensive input in the model** more
expensive. A 960x544 clip at 345 frames is already 52,020 rows, the reference
arms already OOM on 24 GB past about 124 generated frames with images at `max`,
and closing this gap costs roughly 5x what the image one does. It would also be
building the expensive version of an idea whose cheap version is unproven:
settle whether upscaling helps at all on images first. If it does not, the
video question dissolves; if it does, the video fit finally has an evidence base
to justify its cost.

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
  truncates to the generated duration. Costs 80 rows per second of excess;
  core's `TrimAudioDuration` closes it without a new node here.
- **`ref_image_size='max'` does not upscale.** Neither mode does. It picks
  which ceiling sizes a reference down; `MiniMaxH3ReferenceFit` with
  `allow_upscale=True` is the only thing that raises a small one to it.
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
