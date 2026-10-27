> **Part of [`docs/prompting.md`](../prompting.md) (References (`ref2va` Only: Labels, Numbering, Summary & Retention Analysis, §9)).**
> For the core prompt structure (`§1`–`§5.7`) and master router, see [`docs/prompting.md`](../prompting.md).

## 9. References (ref2va only)

### 9.1 The four label types — ref §2, *stated*

| label | meaning |
|---|---|
| `<Subject N>` | visible content abstracted from reference assets that can be reused or modified in the target video |
| `<Picture N>` | a reference image used as a concrete target frame or shot-planning anchor |
| `<Video N>` | a reference video providing an editing source, continuation starting point, or whole-video temporal structure |
| `<Audio N>` | an audio signal that is copied or referenced |

Once a label is assigned, **it keeps the same meaning across all six sections**
(ref §2).

### 9.2 `<Subject N>` — ref §2.1, *stated*

Reusable visible content, covering:

- people, animals, or objects
- scenes, backgrounds, or environments
- clothing, props, interfaces, or visual effects
- styles, actions, expressions, or poses

It is a **content unit that will appear in the target video, not the source
file**. One subject may be defined by multiple assets; one asset may provide
multiple subjects.

```text
<Subject 1> is the young woman in <Picture 1>, with long dark hair, a blue cardigan, and a thin silver necklace.
<Subject 1> is the woman whose appearance comes from <Picture 1> and whose walking motion comes from <Video 1>.
```

### 9.3 `<Picture N>` — ref §2.2, *stated*

Use a **standalone** `<Picture N>` entry only when the image itself serves as a
shot's first frame, keyframe, last frame, edited keyframe, or composition anchor:

```text
<Picture 2> is the first frame of [Shot 1], showing a woman seated beside a café window.
<Picture 3> is a storyboard reference for [Shot 1] and [Shot 2], defining their viewpoint, subject placement, and shot order.
```

**If an image is used only to define a character, scene, costume or style, do not
create a standalone picture entry.** Cite the image source inside the
corresponding `<Subject N>` definition instead. This is the single most-violated
ref2va rule in this repo's shipped prompts.

### 9.4 `<Video N>` — ref §2.3, *stated*

Reserved for **whole-video** relationships: editing an original video, continuing
from the end of one, or referencing its camera movement, cuts, rhythm or temporal
structure.

```text
<Video 1> is the source video for the target video edit.
```

A person, object, scene, action or effect reused from a reference video is still
`<Subject N>`. `<Video N>` identifies the asset or structural source and does not
replace subject labels.

### 9.5 `<Audio N>` — ref §2.4, *stated*

A standalone audio asset, or an enabled synchronised track from a reference
video. Common uses: copying all or part of a signal; referencing a
background-music style; referencing a speaker's timbre and delivery; using
dialogue, lyrics or sound effects from the original audio; referencing beat,
rhythm or continuity.

When an `<Audio N>` corresponds to a target speaker, **reuse that speaker's
global id**: `<Subject N> (Sx)` when the speaker maps to a defined subject, or a
stable voice description followed by `(Sx)` otherwise. The id comes from the
target video's global speaker order and is never assigned or renumbered here.

```text
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).
```

When one audio asset serves multiple roles, describe them in one natural sentence
rather than adding subsections.

### 9.6 Numbering — ref §2.5 (guide), plus the runtime rule (HOUSE)

ref §2.5, *stated*: `<Video N>` and `<Audio N>` are numbered **independently**.
Each index indicates order within its own category and encodes no pairing, so the
same source video may be `<Video 1>` and `<Audio 2>`. An ordinary reference video
does **not** create an `<Audio N>` merely because the file contains sound — only
when its audio is actually used. An `<Audio N>` definition does not have to name
the `<Video N>` it comes from; state the shared source only to remove ambiguity:

```text
<Video 1> is the source video for the target video edit.
<Audio 2> is the synchronized audio track of <Video 1> and is reused in the target video.
```

**The runtime rule the guide cannot tell you** [HOUSE, and enforced]: ComfyUI
emits the labels in **append-chain order**, and a sounded video's `<Audio j>` is
emitted **immediately before its own `<Video k>`**. `<Audio>` is one shared
counter across soundtracks and standalone clips. So a chain of
image, audio, video, audio, video yields:

```text
<Picture 1>  <Audio 1>  <Video 1>  <Audio 2>  <Video 2>
```

and one sounded video plus one standalone clip yields `<Audio 1>` = the
soundtrack, `<Video 1>` = the video, `<Audio 2>` = the standalone. That is not
the order you connected the audio inputs. Get this wrong and the prompt points at
the wrong asset with no warning.

### 9.7 `summary` and the task types — ref §3, *stated*

One short English paragraph summarising the target video and its reference
relationships, beginning with a square-bracketed task-type prefix:

```text
[reference generation] ...
[video editing + reference generation + audio reuse] ...
```

The six legal task types, and nothing else:

| task type | when to use it |
|---|---|
| `keyframe completion` | an image serves as the target's first frame, keyframe, last frame, edited keyframe, or another concrete frame anchor |
| `reference generation` | an image, video or audio asset guides a character, scene, style, action, camera movement, storyboard and so on, **without** serving as a concrete frame or as the source video being edited or continued |
| `video editing` | an existing source video is directly modified; editing an image or generating between still keyframes does not belong here |
| `video continuation` | new content continues, extends, resumes or transitions from an existing source video |
| `audio reuse` | the same audio signal is reused in full or in part |
| `audio reference` | the signal is not copied; only music style, timbre, dialogue or lyric content, sound-effect texture, beat or continuity is referenced |

Rules on combining, all *stated*:

- Combine with ` + ` and **do not repeat a type**. Continuing from a source video
  while using an image as the last frame is `[video continuation + keyframe completion]`.
- **The mere presence of a video or an audio file does not create a task type.**
  A reference video providing only camera movement, cuts or rhythm is normally
  `reference generation`. Use `video editing` or `video continuation` only when
  that video is directly edited or continued.
- When editing a source video, add `audio reuse` if its original audio remains
  audible. When continuing a source video without copying the signal, use
  `audio reference` if the new audio only continues the original's audible
  characteristics.
- **Introduce no new reference labels in `summary`.** Use the ones
  `subject_definitions` already defined.
- For a video-editing task, begin immediately after the prefix with exactly:
  `The target video is an edited version of <Video 1>.`

### 9.8 `retention_analysis` and the relationship markers — ref §4, *stated*

One line per reference label, preserving the meaning established in
`subject_definitions`. **These markers are fixed English values in the output
format**: do not invent, translate, or soften them.

Visible content (`<Subject N>`, `<Picture N>`, `<Video N>`) — ref §4.1:

| marker | meaning |
|---|---|
| `fully_preserved` | the defined role of the referenced content is fully preserved |
| `partially_preserved` | still used, but some defined characteristics are changed or only partially retained |
| `attribute_transfer` | referenced characteristics are transferred to a different identifiable target subject |
| `weak_reference` | only broad similarity in style, category, composition or atmosphere is retained |

Audio (`<Audio N>`) — ref §4.2:

| marker | meaning |
|---|---|
| `fully_copy` | the complete source audio is the target's complete final audio track |
| `partially_copy` | only part of the timeline or selected layers are copied, or sounds are added, removed or replaced after copying |
| `reference` | the signal is not copied; only timbre, rhythm, style, dialogue content or texture is referenced |
| `weak_reference` | only broad similarity in category or atmosphere is retained |

**The two sets share `weak_reference` and nothing else.** A crossed marker is
otherwise a plausible-looking string that means nothing.

Entry forms, from ref §4.1 and §4.2:

```text
<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - ...
<Picture 2> ([Shot 1] first frame): fully_preserved - ...
<Video 1> (cut and pacing structure): weak_reference - ...
<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.
<Audio 2>: reference - the target speaker follows <Audio 2>'s voice timbre and measured delivery without copying the original signal.
```

Two more, *stated*: choose a marker **only within the reference role already
defined for that label**, and **do not treat newly added actions, backgrounds or
plot events in the target video as losses of reference fidelity**.

A hazard worth naming [HOUSE]: all four visual markers presuppose a source. A
subject you invented, with no reference behind it, has nothing to be preserved
from — describe it in `detailed_description` rather than defining it. A house
prompt that filed `weak_reference` on a generated office is what this rule comes
from.

### 9.9 Using the labels in `detailed_description` — ref §5.3, *stated*

At the first clear appearance of an important `<Subject N>`, describe its
referenced characteristics, its position in the frame, and its current action
within what is actually visible in the shot. Keep using the same label later
without redefining it.

Natural phrasings for concrete frame anchors:

```text
the shot begins from <Picture 1>
the shot's keyframe corresponds to <Picture 2>
the shot ends on <Picture 3>
```

Cite `<Video N>` where its source state, structure or continuation relationship
applies, and `<Audio N>` in the shot or semantic phase where the audio
relationship is active.

**Scope this precisely.** ref §5.2 says full-reference mode inserts all four
label kinds in the main field "at their first appearance and where their roles
apply" — but ref §2.2 says an identity-only image gets no standalone entry, and
the guide's own §7 example never cites its four `<Picture N>` labels in
`detailed_description` because all four are identity-only sources. So: cite a
label where its **role** applies. A frame-anchor `<Picture N>` belongs in the
body; an identity-only image belongs inside its `<Subject N>` definition and
nowhere else. **A house rule that bans `<Picture N>` / `<Video N>` / `<Audio N>`
from `detailed_description` outright inverts ref §5.2, §5.3 and §5.4, and makes
`keyframe completion` unexpressible.** The guide wins unless somebody overrides
it deliberately. [OWNER decision territory; the guide reading is GUIDE]

### 9.10 Word budget and detail — ref §5.2 and the preamble, *stated*

- `detailed_description` is **normally 350-500 English words for generation
  tasks**.
- Dialogue-dense content prioritises fitting the complete spoken timeline over
  mechanically reaching a word count.
- Video-editing descriptions scale with the source video's complexity and are
  exempt from the range.
- **A single shot does not automatically justify a shorter description.**
  Distribute detail across shots by information load.
- ref preamble: make `detailed_description` as detailed and explicit as possible.
  For each shot, establish composition, subject appearance and position,
  environment and lighting, actions and state changes, camera movement, current
  sound, and the points where referenced content appears or takes effect. **Avoid
  reducing the description to a plot summary or a list of reference
  relationships.**

This budget is the one ref2va rule most of the composed ref2va arms still miss,
and it is `preflight_graph.py`'s recurring WARN (§13 has the command that lists
them; a count here would be a cache). The two scene arms
(`h3_ref2v_scene_subway`, `h3_ref2v_scene_kitchen`) are written inside it, and
[`scene_arm_renders.md`](../scene_arm_renders.md) says what their renders can and
cannot show.

**Nothing licenses "more words are better".** The one measured length pair,
[`prompt_length_experiment.md`](../prompt_length_experiment.md), is one seed, one
scene and has no working control. What it does license is §9.11's "silence is
not neutral": a long description beat a short one because the short one left a
defective definition in charge, not because it was long.

### 9.11 Reference-writing craft, not in the guide

All HOUSE, all worth having.

- **Treat limiting words such as "only" as strict exclusions.** An explicit
  exclusion blocks incidental transfer. [external writer prompt; the guide
  states what a reference provides and never what it does not, and
  [`h3_references.md`](../h3_references.md) calls the negative clauses open, as
  §15.6 does negative constraints in general]

  | phrase | transfers | not |
  |---|---|---|
  | "room only" | architecture, layout, surfaces, fixtures, furnishings | people, actions, wardrobe, camera, lighting, style, audio |
  | "clothes only" | clothing design | identity, body, pose, environment, medium |
  | "body shape only" | body proportions | face, hair, clothing, pose, style |
  | "face only" | facial identity | body, clothing, pose, environment, medium |
  | "camera only" | camera behaviour | subjects, setting, action, style, soundtrack |
  | "movement only" | motion or choreography | identity, clothing, environment, camera, audio |
  | "voice only" | voice qualities | words, music, ambience, other speakers |
  | "style only" | visual treatment | source identity, objects, setting, composition, action |

- **Bind each attribute to its source.** A subject can take different
  attributes from different sources; say which comes from which, as in
  "`<Subject 1>` is the woman whose identity and body shape come from
  `<Picture 1>` and whose clothing design comes from `<Picture 2>`". Import
  nothing unassigned. [external writer prompt]
- **Wardrobe grammar.** Resolve a clothing instruction into one of four
  operations, then state the final visible outfit **once** and do not repeat a
  competing description later:
  - "same clothes" preserves the established outfit;
  - "wearing an apron" normally **adds** it over the outfit;
  - "replace her clothes with jeans and a T-shirt" **replaces** the outfit;
  - "make the top red" **modifies** only the specified colour;
  - "use the jacket from `<Picture 2>`" transfers **only** that garment.

  Preserve plausible layering, fit, attachment and occlusion.
- **Cast sheet.** Before writing, resolve every named character, franchise
  character or real person into appearance (species or body type, height and
  build, palette, silhouette, face, costume down to named garments and colours,
  one distinguishing feature) and voice (on-screen or off, pitch, timbre, rate,
  accent). Keep the name **and** give the description on first mention, then
  reuse the **same noun phrase verbatim** on every later mention. H3 renders what
  is described, not what is named: a bare proper noun comes back as a
  plausible-adjacent stranger. Naming a source property as a *style* anchor is
  worth doing; naming it as a substitute for description is not. A likeness of
  an actor's character has also been introduced once, in shot 1, as `[Name]
  (played by [Actor] in [Show])`, with the vocal timbre in the prose outside
  `<d>`: external writer craft that the `h3-prompt` skill carried until
  2026-10-01, unmeasured here, and it does not replace the description. [OPEN]
- **Silence is not neutral.** Two observed results, one render each, neither in
  any guide: `subject_definitions` beat `retention_analysis` (a brunette
  reference defined as blonde rendered blonde under `fully_preserved`), and a
  specific `detailed_description` beat `subject_definitions` (a definition that
  claimed architecture for a lakeside reference drove the render only where the
  description said nothing about the environment). So **describe the
  environment in `detailed_description` even when a reference supplies it**, and
  do not assert an attribute you have not looked at. The same mechanism is §16.
  [HOUSE, observed; [`h3_references.md`](../h3_references.md) "A label is a bare
  ordinal" and [`prompt_length_experiment.md`](../prompt_length_experiment.md)]
- **Conditioning, not instruction.** The prompt reaches the encoder as written
  (§15.2); nothing on the other side decides whether to comply, and there is no
  negative prompt field. "Make it feel tense" conditions on the words "make it
  feel tense". "Her knuckles whiten on the railing" conditions on an image.
- **A change of medium keeps the assigned design.** "Make it real" or "render
  as live action" changes presentation, not identity: keep every attribute the
  source is assigned (face, body shape, hair, clothing design, colours,
  markings, proportions) and translate it into the target medium, dropping
  source-medium artefacts such as illustration lines, paper texture, toy seams
  or plastic surfaces unless asked. Such a conversion can still be
  `fully_preserved`, because ref §4 defines that as the reference's *defined
  role* surviving (§9.8). [HOUSE reading of a GUIDE term, external writer
  prompt]
- **The prompt is self-contained, in every mode.** The model receives only the
  prompt and the wired media, so a phrase pointing anywhere else points at
  nothing: "as described above", "as previously mentioned", "based on the
  user's description", "the requested outfit", "same as before". Resolve each
  into the content itself. [external writer prompt]

---

