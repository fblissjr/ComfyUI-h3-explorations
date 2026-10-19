# Prompting references: what a label means, and each relationship

The detail under [`references.md`](references.md), which owns the label rules (what the tokenizer emits and in what
order). The prompt format itself, its six sections, its marker vocabulary and the writing rules, is owned by
[`../prompting/prompting.md`](../prompting/prompting.md): sections 2.1 and 2.2, 9.1 to 9.4 and 9.8, and 9.11,
"Silence is not neutral". This page says what follows for references, and gives one worked prompt per relationship.

## A label is a bare ordinal: the prompt has to say what it is

The tokenizer emits `<Picture 1>`, `<Picture 2>`, `<Video 1>` and nothing else. No socket, flag or payload field says
what a reference is, what it is for, or which subject it belongs to (verified against `comfy/ldm/minimax/model.py`).
Three stills wired to one subject and three wired to three subjects are identical graphs. **The prompt is the only
place the inputs acquire meaning.** Measured 2026-08-16 in a paired render
([`../prompting/prompt_length_experiment.md`](../prompting/prompt_length_experiment.md) has the pre-registration and
verdict).

**The authority order, so far** (two results at one render each, same direction, in no guide):

1. **`subject_definitions` beats `retention_analysis`.** A brunette reference described as blonde in the definitions
   rendered blonde, despite `fully_preserved` on the retention line (the owner's observation).
2. **A specific `detailed_description` beats `subject_definitions`.** Both arms carried a definition claiming
   `<Subject 2>` had "architecture", against a mountain-lake reference with no buildings. The arm whose description
   said nothing about the setting put the subject in a timber veranda beside a chalet; the arm that named the lake,
   reflection, meadow and conifers rendered no structure.

So a wrong word in `subject_definitions` is load-bearing exactly as far as nothing downstream contradicts it. The
length arm had no working control (the shared camera sentence was elaborated differently in the long arm), so the
direction is credible and the magnitude is not bounded.

**What follows for writing one:**

- Say what each reference is, do not assert an attribute you have not looked at, and describe the setting even when a
  reference supplies it (`prompting.md` 9.11). The image-reference template says "setting" rather than "architecture,
  palette, and lighting", because `_ref_prompt()` cannot see the image.
- A reference wired but never described costs its rows every step and says nothing.
  `bench/check_ref_prompt_labels.py` catches the unnamed case; `bench/preflight_graph.py` also warns when a defined
  label is never cited in `detailed_description`, which is the guide's requirement
  (`coderef/MiniMax-H3/skills/h3-prompt-writing/references/ref-en.txt:231`). Both prompt instruments read the guide
  from the graph and support native sockets and typed chains (`bench/check_prompt_guide_conformance.py`); neither
  checks that the marker chosen is sensible.

## The relationships, and how to ask for them

Which sockets are wired is mechanical; **what the prompt asks the labels to do is what changes the output.** The
official guide names three whole-video relationships plus a subject-sourcing rule that yields a fourth.

### Edit a source video (the closest thing to inpainting)

**There is no mask.** It is whole-frame regeneration conditioned on the source; what holds the untouched parts is
`retention_analysis` saying precisely what survives.

The pack's masked path is a different route: there the source is the song node's starting latent, not a reference,
and the plate is kept exactly, but it is not a trained task
([`../masked_v2v/overview.md`](../masked_v2v/overview.md); [`audio_freeze.md`](audio_freeze.md) section 4). This one is
trained and keeps nothing exactly.

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

**The vendor's serving path has no edit task** (read 2026-10-08, sglang `214347891a`). An edit there is this same request: a reference video and
a prompt that says it is the source. Its tasks are text, keyframes
and references (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py::MINIMAX_H3_TASK_PROFILES`),
a condition is a keyframe or a reference and carries nothing else
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py::_ALLOWED_CONDITION_KEYS`),
and nothing in its own pipeline starts from the source's latent, masks it
or anchors a frame from it. What follows for anyone reading an edit's
render:

- **The source's frames are not at the target's time positions.** Every
  reference block is placed before the target on the time axis, a video
  block with its soundtrack at one shared origin, and the target starts
  where the last block ends
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py::minimax_h3_packed_sequence_ref2va_blocks`;
  core's `comfy/ldm/minimax/model.py::PackedLayout` does the same). Frame
  for frame correspondence is something the model does through attention,
  not something the layout states.
- **In space the source does lie over the target when the two have one
  aspect ratio.** A frame's rows are spread over a span set by its aspect
  ratio alone, at any resolution
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py::_axis_from_sqrt_area`,
  and `comfy/ldm/minimax/model.py::_axis_from_sqrt_area`), so a source at
  the canvas's shape covers the same coordinates as the target, and one of
  another shape does not.

[`docs/research/sglang_comparison.md`](../research/sglang_comparison.md), "Ninth
read", has the rest of that read.

**When the source already shows the person, leave the still out (owner,
2026-10-09).** A reference still is there to give the model a face. A source
video that already shows that face gives it too, provided the video model
has its own copy of the source: `use_vae` on the reference video, or
`motion_vae` on a `MiniMaxH3MaskedSource` whose mask covers the whole frame.
In that case wire no still: one input fewer, and its rows with it. Wire the
still when the person is not in the source, which is the ordinary first
pass, or when the source reaches the text encoder only, where the lead came
out as someone else. This rests on stills from two renders of one clip
against one, not on playback, and no default moved:
[`bench/results/2026-10-09_whole_frame_text_and_copy_size.md`](../../bench/results/2026-10-09_whole_frame_text_and_copy_size.md),
"The lead without a still".

### Replace a character, keeping the video as the plate

The same sockets as the edit, the opposite question: the person is the only thing that changes. `<Video 1>` is the
plate; `<Picture 1>` is the new identity. Graph: `h3_ref_video_swap_api.json`, the one swap graph.

```
subject_definitions:
<Subject 1> is the character whose complete visual identity -- face, facial structure, eyes, skin tone, hair style and colour, body proportions, and overall appearance -- comes exclusively from <Picture 1>. Their body motion, posture, gestures, head movements, timing, and physical performance come from the original character in <Video 1>.
<Picture 1> supplies subject identity only. It does not supply lighting, exposure, colour grade, background, camera angle, pose, framing, or scene composition.
<Video 1> is the source video for the target video edit. It supplies the camera path, framing, background, environment, lighting, composition, action timing, and the original character's body motion. It does not supply the face or identity.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - facial structure, identity, hair, and appearance from <Picture 1> are retained.
<Video 1> (environment and motion): partially_preserved - the setting, lighting, and camera composition are retained, and the original character's actions are transferred to <Subject 1>.
```

- **The negative clauses are not from the official guide**, which only ever states what a reference provides. They
  come from general prompting research, against the reported failure of blended identities or the still's lighting
  pulled into the plate. An open question, not a rule.
- **One still, not two**: the plate supplies the setting, and a second reference would pay rows every step to say
  nothing. When likeness fails, rule out a low-resolution reference or a face far from the camera first.
- **Speech in a reused soundtrack degrades often at this canvas and length, and the prompt is not shown to be the
  cause.** Of four renders on 2026-08-22 (structured at two seeds, a concise twin and an imperative arm at one each,
  1152x768, 124 frames) the owner found one clean, and the other structured render was not, so the split does not
  follow the prompt; the imperative arm was deleted on the owner's call. A later batch at the shipped canvas and
  length broke the concise twin's speech on every seed, by ear and by `bench/grade_soundtrack_reuse.py`
  ([`../data/results/2026-08-22_swap_prompt_verdict_362.json`](../../data/results/2026-08-22_swap_prompt_verdict_362.json);
  CHANGELOG 0.57.0), and it was retired; structure, length and how the soundtrack is stated moved together there too.
  Ruled out for that batch: frame-rate desync (the 25 fps source is loaded with `force_rate=24`, and VHS's audio window
  covers the same span as its frames) and span expansion (both windows were 5.167 s, and 124 is on the grid). Still
  open: the owner's hypothesis that 124 frames is too short for the source's dialogue density, which a length sweep on
  one prompt would test.

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

## The shipped reference graphs

The first six vary which sockets are wired; the last five vary what the prompt asks for, holding the wiring roughly
still. All load the `ref2va` checkpoint.

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


The `h3_probe_ref_pathway_*` family ([`references.md`](references.md), "Encoder-only references") is
the exception: two of its arms wire core's `MiniMaxH3ReferenceToVideo`
rather than the typed chain, and one loads `fl2va`.
