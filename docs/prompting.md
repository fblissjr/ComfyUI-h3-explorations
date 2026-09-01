# How to write an H3 prompt

last updated: 2026-09-01

**The single source of truth for writing an H3 prompt, in any mode.** Everything
needed is restated here: the closed vocabularies in full, the exact Part One
templates, the section layouts, all seven markers, graded worked examples
per mode, what the model actually receives, and where every source that
claims to govern a prompt disagrees with the others. **You do not need
`internal/` and you should not need any other file.**

`internal/PROMPTING.md` is superseded and being retired into this file; §14.5
says what has moved and what has not. §14 is the reconciliation across all five
sources — read it before citing any of them, because two are not authorities.

Companions, neither of which is a source for the rules themselves:
[`prompt_catalogue.md`](prompt_catalogue.md) is what we currently render
(generated from the graphs); [`prompt_audit.md`](prompt_audit.md) is whether
those follow these rules.

## Four layers, and every rule says which one it is

Collapsing these is how two invented rules shipped here and were retracted.

| layer | what it is | breaking it means |
|---|---|---|
| **GUIDE** | the vendor's own text, `internal/official_prompt_guides/` base_en and ref_en, cited by section | the prompt is **off-distribution** from what the model was trained on |
| **OWNER** | a deliberate design decision made here | not the vendor's, not a defect; cannot be cited as authority |
| **HOUSE** | our inference from measurement or experience | may itself be wrong; check before relying |
| **OPEN** | contradictory, unsourced, or unverifiable | do not build a checker on it |

**The guides are the only authority, and the line between a guide STATEMENT and
a guide EXAMPLE is load-bearing.** Two rules have been invented here by reading
the guides' worked examples as if they were rules, and both were retracted. Every
GUIDE rule below is marked *stated* or *shown* accordingly. A *shown* rule is
still worth following, because following the vendor's examples keeps you
in-distribution; it is not worth enforcing, and it is not worth arguing from.

Two guides exist and they do not share a section list:

- **base_en** governs T2VA, I2VA, L2VA and FL2VA. Three core fields.
- **ref_en** governs full-reference mode (ref2va). Six sections. It inherits
  base_en's shot, camera, speaker, dialogue and sound formats by reference
  (ref §5.1) and overrides four things (ref §5.2).

---

## 1. Pick the mode first

base §1 defines the four base tasks; ref_en covers the fifth.

| mode | what you supply | what the model is asked to do |
|---|---|---|
| **T2VA** | text only | build the whole timeline from text (base §1) |
| **I2VA** | a first frame | develop forward from that frame (base §1, §3.1) |
| **FL2VA** | a first and a last frame | a continuous path from the first to the last (base §1, §3.2) |
| **L2VA** | a last frame | converge from a plausible earlier state onto that frame (base §1, §3.3) |
| **ref2va** | reference images, videos, audio | generate with those assets driving identity, scene, style, motion, camera or voice (ref_en) |

**An image that is literally frame 0 (or the final frame) of the output is a
keyframe, not a reference.** [HOUSE] Keyframes go through the base guide's
alignment sentence; references go through ref_en's label system. ref_en's
`keyframe completion` task type is the case where one asset is both, and it is
only claimable on a graph that wires a keyframe node.

---

## 2. Structure, per mode

### 2.1 T2VA, I2VA, L2VA, FL2VA: Part One then three core fields

base §2.2 gives the three core fields, in this order, separated by a blank line:

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

- **`integrated_multimodal_description`** carries visuals, actions, shots,
  speakers, dialogue, singing and diegetic audio along the timeline (base §2.2).
- **`overall_soundscape`** summarises ambience, physical action sounds and
  non-verbal human sounds across the whole video (base §2.2, §4.6).
- **`non_diegetic_music`** is background music the characters cannot hear
  (base §2.2, §4.7).

Part One is the image-alignment instruction. base §2.1, *stated*: it **must be
the first line of the final prompt, followed by one blank line before the core
fields**, and T2VA has none at all.

**The three templates are not interchangeable and differ in punctuation as well
as wording.** Reproduce them character for character, em dash included.

**T2VA** — no Part One. The prompt begins directly with
`integrated_multimodal_description:`.

**I2VA** — base §2.1, *stated*, "always uses":

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
```

**FL2VA** — base §2.1, *stated*, "always uses":

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.
```

**L2VA** — base §2.1, *stated*, "always uses":

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.
```

**The bracket convention is the trap.** FL2VA is the one variant that carries
**no angle brackets and no square brackets**: bare `Picture 1`, bare `Shot 1`.
I2VA and L2VA bracket both. A writer who "normalises" FL2VA to `<Picture 1>
(from [Shot 1])` has left the string the guide says the mode always uses.

**Resolving the placeholders** (base §2.1, *stated*): `N` is the index of the
actual final shot, and `S.SS` is the effective video duration formatted to
**exactly two decimal places**. For FL2VA and L2VA on a single-shot prompt, `N`
is 1 — which is what the guide's own Cases 3 and 4 show.

**Inside the body**, the same bracket convention carries through: base §3.1 and
§3.3 write `<Picture 1>` in prose for I2VA and L2VA, while base §3.2 and Case 3
write bare `Picture 1` / `Picture 2` for FL2VA. This is *shown*, not *stated*.
Follow it; do not argue from it.

### 2.2 ref2va: six sections

ref §1, *stated*, gives six sections in this order:

| section | purpose |
|---|---|
| `subject_definitions` | defines referenced content and its reference labels |
| `summary` | task type, target video, main reference relationships |
| `retention_analysis` | how referenced content is preserved, transferred or reused |
| `detailed_description` | visuals, actions, shots, sound, dialogue in playback order |
| `overall_soundscape` | ambience and physical sounds |
| `non_diegetic_music` | background music audible only to the audience |

There is no Part One in ref2va. The alignment sentence is a base-guide
construct; ref_en's equivalent is `subject_definitions` plus `retention_analysis`.

**Write all six sections in English** (ref, preamble, *stated*). The original
language is preserved only for dialogue and lyrics inside `<d>` and for text
visibly present in the scene.

### 2.3 The four ref2va differences from the base format

ref §5.2, *stated*, is a differences table. Learn all four; three of them are
places house prompts have gone wrong.

| dimension | base modes | ref2va |
|---|---|---|
| main field | `integrated_multimodal_description` | `detailed_description` |
| style opening | written **after** `[Shot 1]` (base §4.1) | one or two English sentences **before** `[Shot 1]` |
| reference labels | not used | `<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>` inserted at first appearance and where their roles apply |
| audio | describes the video's own sound | cites `<Audio N>` in the corresponding shot or audio phase and states whether the signal is copied or referenced |

ref §5.2's opening example:

```text
The target video is in a cinematic, literary music-video style with soft lighting and a slightly desaturated color palette.
[Shot 1] The scene opens in a crowded urban street...
[Shot 2] At 00:09.000, the shot cuts to an extreme close-up...
```

### 2.4 Where the section label sits on the line: ambiguous, do not enforce

base §2.2's structure block and all four base Cases put the content on the
**same line** as the label (`integrated_multimodal_description: [Shot 1] ...`).
ref_en is inconsistent with itself: its §7 complete example puts each label on
its own line with content starting on the next, while its §6 snippets are inline
(`overall_soundscape: Quiet indoor room tone...`).

Neither guide **states** a rule about this. Both positions are example-derived,
and ref_en's examples disagree. Treat layout as OPEN, write whichever you like,
and do not "correct" a prompt on this axis. What *is* stated: entries inside
`subject_definitions` get one line per item (ref §2) and entries inside
`retention_analysis` get one line per reference label (ref §4).

---

## 3. Shots and timing

### 3.1 Shot headers

base §4.2 and ref §5.1, *stated*:

- **`[Shot 1]` carries no timestamp.** Do not add one.
- Later shots use sequential numbers and open with a **strictly increasing** cut
  time that **falls within the video duration**.
- The format is `[Shot N] At MM:SS.mmm, ...` (spelled out in ref §5.1; base §4.2
  shows it: `[Shot 2] At 00:03.500, the camera cuts to...`).

A malformed header such as `[Shot 1, 00:00.000-00:06.000]` is not merely
non-conformant: `bench/preflight_graph.py` requires a literal `]` after the
digits, so its shot list comes back empty and **three shot rules go inert
silently**. The grader is removed rather than reddened.

### 3.2 Cuts

base §4.2, *stated*. For an ordinary cut, use one of exactly these five:

```text
the camera cuts to
the shot cuts to
the shot transitions to
the shot changes to
the shot switches to
```

`cross-dissolve`, `fade` and `wipe` are permitted **only when the user
explicitly requests one**.

**When to cut** (base §4.2, *stated*): a cut should introduce new information
about the subject, space, state, viewpoint or time. If only the distance or a
slight angle needs to change, **prefer camera motion**.

### 3.3 Duration and the frame grid

Neither guide says anything about frame counts. This section is HOUSE and
runtime-derived, and it matters because the FL2VA and L2VA alignment sentences
demand a duration to two decimals that nothing downstream will correct.

ComfyUI snaps `length` **up** to the next `n = 17k + 5`, at 24 fps
(`h3_rules.py::snap_length`, matching `align_frame_count` in core). Achievable
durations are therefore quantised:

```text
90=3.750  107=4.458  124=5.167  141=5.875  158=6.583  175=7.292
192=8.000  209=8.708  226=9.417  243=10.125  260=10.833  277=11.542
294=12.250  311=12.958  328=13.667  345=14.375  362=15.083
```

For any other frame count, duration = n / 24. **192 frames (8.000 s) is the only
common integer duration that lands exactly**; "10 seconds" is 10.125 at 243
frames and 10.833 at 260.

**Duration honesty** [HOUSE]. Only write a snapped duration into an alignment
sentence when you were given a frame count. If you were given seconds, you do not
know which frame count will be typed, so use the requested value and say that it
is a request rather than a grid position.

**State the duration in the prose too** [HOUSE, from a vendor example]. MiniMax's
own FL2VA material writes "throughout the entire eight-second duration". It costs
six words and gives the model a second signal about how long it has.

### 3.4 Timing margins

All HOUSE or third-party, none in either guide. Believe them less strongly than
anything marked GUIDE.

- Last cut no later than `duration - 2.5 s`. The final beat is the one that gets
  squeezed, so put the shot you care about in the middle.
- No speech before about `0.4 s`; the last line ends by `duration - 0.8 s`.
- **Speech budget**: `max_words_in_shot = 2.5 x (shot_seconds - 1.0)`. The
  functional form is **assumed, not measured** — a one-parameter `2.2 x s` fits
  the single anchor (a vendor 5 s example carrying 11 spoken words) exactly where
  this one is off by one, and the two diverge by 1.8x at 2 s. Treat short-shot
  budgets as possibly far too tight. [OPEN]
- **Turn cap: OPEN.** House material states both "about one speaking turn per 3 s"
  and "no cap beyond the word budget", using the same five-second clip as the
  example. Neither guide says anything about turn counts. Do not enforce either.

---

## 4. Camera motion: the closed vocabulary

base §4.3, *stated*. A complete expression has three dimensions and **all three
draw from closed sets**. Anything outside them is off-distribution.

| dimension | value | meaning |
|---|---|---|
| motion type | `Zoom In / Zoom Out` | focal length changes, camera body stationary |
| motion type | `Push In / Pull Out` | the camera moves forward / backward |
| motion type | `Pan Left / Pan Right` | camera in place, lens pivots horizontally |
| motion type | `Truck Left / Truck Right` | the camera translates horizontally |
| motion type | `Tilt Up / Tilt Down` | camera in place, lens pivots vertically |
| motion type | `Pedestal Up / Pedestal Down` | the whole camera moves up / down |
| motion type | `Arc Shot` | the camera moves in an arc around the subject |
| motion type | `Tracking Shot` | the camera follows a moving subject |
| motion type | `Static Shot` | camera position and lens remain still |
| motion type | `Shake Slightly / Shake Strongly` | slight / strong camera shake |
| motion type | `POV` | the subject's point of view |
| motion type | `Roll Clockwise / Roll Counterclockwise` | the camera rolls around the lens axis |
| amplitude | `with small amplitude` | small-range change |
| amplitude | `with large amplitude` | large-range change |
| speed | `at slow speed` | slow movement |
| speed | `at fast speed` | fast movement |

**Add amplitude and speed only when they are meaningful**; base §4.3 says medium
amplitude and normal speed are usually omitted.

**Write it as natural English action inside the shot**, not as labels stacked at
the end of a sentence. base §4.3's own examples:

```text
The camera pushes in with small amplitude at slow speed toward the folded letter in her hands.
The camera pans right with large amplitude at fast speed, revealing the open doorway.
The camera holds a static shot as the runner exits the frame.
```

### Three traps, all of which have shipped here

- **Medium and normal are not expressions, they are omissions.** `at medium
  amplitude and moderate speed` is wrong twice: the words are not in the table,
  and the concept is "write nothing".
- **`Truck Left` and `Tracking Shot` are different rows.** `tracks left`
  conflates them.
- **A `[Shot N]` header carrying a timestamp already IS the cut.** `whip pan` is
  absent from the table entirely, and in the prompt where it shipped it also
  re-described a cut the header had already made. Write the cut, then the move.

Two more, house-level and useful: **H3 reframes by default**, so to hold a frame
name the moves that do not happen as well as asking for a static shot; and
**one primary change per beat**, because two changes collapse into whichever is
easier to render. [HOUSE, third-party]

---

## 5. Speakers, dialogue and singing

base §4.4 governs all modes; ref §5.1 inherits it and ref §5.4 adds the
reference-specific rules.

### 5.1 Speaker ids

base §4.4, *stated*:

- Subjects who **speak, sing, or produce an off-screen human voice** use stable
  ids such as `(S1)` and `(S2)`.
- A speaker keeps the same id across shots.
- **Characters who never vocalise receive no speaker id.**
- When **multiple already-numbered speakers** speak or sing together, use a
  compound id such as `(S1,S2)`.

Two consequences worth spelling out. **Speaker ids are not conditional**: the
guide states them unconditionally and every guide example carries them, so a
system prompt that makes them optional for a single unambiguous speaker has left
the format. And **a compound id instructs both speakers to say the line in
unison**; because both must be *already-numbered*, you cannot introduce a speaker
with one, and it is never a way to mention a second character. [the "already
numbered" wording is GUIDE; the unison reading is HOUSE]

### 5.2 Where identity goes

base §4.4, *stated*: **when a speaker first appears**, provide enough information
from the visual and audio context to establish a stable identity — character
type, age, gender, whether the person is on-screen, pitch, timbre, speaking rate,
accent.

Note "first **appears**", not "first speaks". Introducing `(S2)` in one shot and
describing them in the next is a violation, and it is a live defect in a shipped
house prompt.

### 5.3 The `<d>` block

base §4.4, *stated*: place the speaker's identifying phrase, id, action and
delivery **outside** `<d>`. **Inside `<d>`, include only the language tag and the
actual user-provided spoken content.** Preserve every original word and
punctuation mark verbatim; do not translate or rewrite them.

The form is `<d>[Language] ...</d>` (ref §5.1, *stated*). base §4.4's examples:

```text
The young woman with a quiet, breathy voice (S1) says: <d>[English] I get off at the next station.</d>
The two children (S1,S2) shout together, <d>[English] Wait for us!</d>
```

Neither guide lists which languages are supported. A house list circulates
(Arabic, Chinese, English, French, German, Italian, Japanese, Korean,
Portuguese, Russian, Spanish); it is **not** in either guide. [HOUSE]

### 5.4 Voiceover

base §4.4, *stated*: use the **exact phrase** `says in an off-screen voiceover`.
Immediately after every voiceover `<d>` block, state that the corresponding
on-screen character's lips remain closed.

```text
The man (S1) says in an off-screen voiceover: <d>[English] I still remember that road.</d> while his lips remain completely closed.
```

### 5.5 Dialogue across a cut, and speech cut off by the end

base §4.4, *stated*: when the same line of dialogue or lyrics crosses a cut, use
`<scenetrans>` at the connecting points in **both** parts and explicitly state
that the audio continues across the cut. Use `<cutoff>` when speech is truncated
by the end of the video.

The four continuity phrases the guide names:

```text
continues seamlessly across the cut
continues uninterrupted into the next shot
carries over from the previous shot
remains audible across the transition
```

**Both spellings are a problem, and this is the sharpest guide-versus-release
conflict in the format.** The release declares `<|cutoff|>` with pipes and
declares no `<scenetrans>` at all (see §7). Both guides print `<cutoff>` and
`<scenetrans>` unpiped. House practice is to write the piped `<|cutoff|>` and to
treat `<scenetrans>` as ordinary prose that matches the guide's wording.
[GUIDE and release disagree; the resolution is HOUSE]

### 5.6 Closing the mouth

Not in either guide as a general rule, but present in the vendor's own worked
material and worth doing on every line: at the moment a line ends, describe the
lips closing and the jaw ceasing to move, or the mouth keeps moving past the
audio. [HOUSE, from a vendor example]

Two companions, both HOUSE:

- Every on-screen character who does **not** speak gets an explicit "produces no
  vocal sound", or the model may voice them.
- **L2VA endpoint rule**: if the final picture shows a closed-mouth expression,
  finish the dialogue early enough for the mouth to return to that expression
  before the endpoint. A closed-mouth expression cannot remain physically
  unchanged during speech.

### 5.7 ref2va additions

ref §5.4, *stated*:

- When a referenced subject physically speaks, keep **both** labels:
  `<Subject 2> (S1) turns toward the woman and says, <d>[English] ...</d>`
- Off-screen, same form, marked `off-screen`.
- When the speaker corresponds to no defined subject, use a **stable voice
  description** followed by `(Sx)`.
- Assign `(Sx)` **once**, in the order of actual vocal events in the target
  video, and reuse it at every vocal event. An `<Audio N>` definition bound to a
  target speaker reuses the same id and never assigns a new one.
- **Never write `(Sx)` in `retention_analysis`.**
- When verbal content exists only inside a directly reused BGM or complete
  soundtrack and **no person, character, narrator or other independent vocal
  source produces it**, cite `<Audio N>` as the audible source and do **not**
  invent an `(Sx)`. If a concrete vocal source produces it, assign and reuse
  `(Sx)`.

```text
When <Audio 1> reaches the phrase <d>[English] I'm lonely lonely lonely lonely lonely I'm lonely</d>, <Subject 1> performs the corresponding hand gesture without becoming a separate speaker source.
```

**Reused source dialogue** (ref §5.4, *stated*, and this is the only place the
punctuation rule lives): when dialogue, narration or lyrics from reference audio
are directly reused, or when the input explicitly requests their reperformance,
preserve the exact source words and original language inside `<d>`. Write
`[unclear]` for unintelligible spans rather than guessing. Standardise
punctuation to `,` `.` `?` `!`, remove repeated tildes, emoji, bullets and
decorative punctuation, and end complete statements, questions and exclamations
with `.`, `?` or `!` before `</d>`.

**That closing-punctuation rule is scoped to reused source dialogue and nothing
else.** Applied universally it collides head-on with base §4.4's "preserve every
original word and punctuation mark verbatim". A house rule that states it
universally is over-reaching. [OPEN]

When only timbre, rhythm, emotion or delivery is referenced, do **not** carry the
original dialogue into the target video (ref §5.4, *stated*).

---

## 6. On-screen text

base §4.5, *stated*: place any **banner, sign, label, subtitle, or neon text**
that is actually visible on screen in **English double quotation marks**.
Preserve the original text and punctuation verbatim, without translation.

```text
A red neon sign reading "营业中" glows above the doorway.
```

Two things follow. **Subtitles are on-screen text**, and this is the documented
route to one: the guide lists subtitle in the same breath as signage and sends
all of it through the double-quoted string. And **an unspecified string renders
as letter-shaped noise** — a HUD described only as "HUD elements" came back as
`ETR METNO CITFEP` in third-party testing, so name the typography and where it
sits in frame as well as the string. [the quoting rule is GUIDE; the noise
failure mode is third-party HOUSE]

Burned-in subtitles are known to render at all: a colleague's clip came back
with five legible lower-third subtitles from a prompt using the §4.5 prose form.
That establishes the capability, not which mechanism produced it — the prompt
was not captured, so a caption-marker route is not excluded. [HOUSE, one clip]

---

## 7. Markers: all seven, and the two the guides name

The release declares exactly seven H3 special tokens in
`additional_special_tokens` (`vendor_config/tokenizer_config.json`):

```text
<d>  </d>  <|cutoff|>  <|lyrics_start|>  <|lyrics_end|>  <|caption_start|>  <|caption_end|>
```

**The guides document two of them** (`<d>` / `</d>`). Everything below about the
other five is HOUSE pattern, and the structural rules are enforced by
`bench/preflight_graph.py` rather than derived from any guide.

| marker | what it is for | where it goes |
|---|---|---|
| `<d>` … `</d>` | spoken or sung words | inline in the main field; language tag plus the words, nothing else (GUIDE, base §4.4) |
| `<\|cutoff\|>` | speech truncated by the end of the video | directly against the closing `</d>`, no space (HOUSE; the guides spell it `<cutoff>`) |
| `<\|lyrics_start\|>` … `<\|lyrics_end\|>` | marks a sung run | **wraps** one or more `<d>` blocks (HOUSE) |
| `<\|caption_start\|>` … `<\|caption_end\|>` | undocumented; the name reads as subtitle | a **sibling** of `<d>`, immediately after the `</d>` it belongs to (HOUSE) |
| `<scenetrans>` | dialogue continuing across a cut | at the connecting point in **both** parts (GUIDE, base §4.4) — **but it matches no declared token** |

Structural rules, all HOUSE and all checked:

- Every pair opened is closed.
- **The lyrics pair is the only thing that ever wraps a `<d>`.** A caption pair
  never wraps one, and no marker pair ever sits inside a `<d>`.
- No marker pair nests inside another marker pair.
- No whitespace padding inside a caption pair; the padding is part of the string.
- No whitespace before `</d>`.
- Markers ride inline in the shot prose. Only `[Shot N]` starts a line.
- One caption pair per on-screen line; two adjacent pairs are one line split in
  two.
- No full stop directly before `<|cutoff|>`. **The rule is now unsourced rather
  than mechanical** [OPEN]: its stated mechanism, BPE dragging the `.` into the
  marker's leading fragment, described a pre-fix tokenizer path that native
  ComfyUI no longer takes.

**What `<|caption_start|>` means is undocumented.** It appears in neither guide,
in no vendor script, and in no worked example anywhere; its name and its position
in the declared list beside `<d>`, `<|cutoff|>` and the lyrics pair are the only
evidence. An earlier house reading of it as signage was withdrawn. Whether it
renders anything is unmeasured; a controlled marker-on/marker-off pair at matched
seed has been designed and not completed. **Use base §4.5's double-quoted string
as the primary route to a subtitle and treat the marker as an addition, never a
replacement.** [OPEN]

**Marker rows are untrained, and that discriminates nothing.** All seven sit with
the untrained padding tail of the encoder's embedding table — but the released
text encoder is byte-identical to stock Qwen3-VL-32B-Instruct, so every row is
untrained-by-MiniMax and the seven could not have been otherwise. Do not read
"untrained" as "do not use them", and do not read it as evidence about how
MiniMax tokenised them.

---

## 8. The two audio fields

Shared by every mode. base §4.6 and §4.7 own them; ref §6 defers to those
definitions and adds the reference-audio rules.

### `overall_soundscape` — base §4.6, *stated*

- **1 to 4 English sentences in one continuous paragraph.**
- Ambient sound, physical action sounds, non-verbal human sounds across the full
  video: wind, rain, traffic, footsteps, fabric movement, impacts, breathing,
  laughter, panting.
- Dialogue, singing and diegetic music belong in the main field and **must not**
  be repeated here.
- `N/A` **only** when the user explicitly requests complete silence throughout.

```text
overall_soundscape: Steady rain taps against the café windows while low room ambience continues underneath. The entrance bell rings once, followed by wet footsteps and the soft scrape of a chair.
```

**Two things that are NOT rules here**, recorded because both were briefly
written down as rules and retracted. §4.6 constrains only "1-4 English sentences
in one continuous paragraph" and nothing about their internal shape: sequenced,
chronological prose appears in every worked example and is stated nowhere.
And §4.6 puts **physical action sounds in the soundscape by name** — coins,
impacts and footsteps belong there, and its "should not be repeated here" covers
dialogue, singing and diegetic music only.

### `non_diegetic_music` — base §4.7, *stated*

- **1 to 3 English sentences.**
- Background music the characters cannot hear and only the audience can hear.
- Focus on **instrumentation, speed, rhythm and dynamic changes**. Do **not** use
  abstract mood words or explain the emotional function of the score.
- Singing, instruments, radio, television or phone music audible to the
  characters are **diegetic** and belong in the main field.
- `N/A` when there is no non-diegetic music.

```text
non_diegetic_music: Sparse piano notes at a slow tempo, joined by sustained low strings that gradually increase in volume before fading out.
```

### ref2va additions — ref §6, *stated*

State a copy or reference relationship **only in the section that matches the
audible layer**: ambience and sound effects in `overall_soundscape`,
audience-only score in `non_diegetic_music`. If one asset supplies both, describe
the corresponding relationship in each.

```text
overall_soundscape: The copied ambience layer from <Audio 1> continues throughout the target video.
non_diegetic_music: <Audio 2> is directly reused as the complete audience-only score.
```

Write complete dialogue and lyrics **only** inside `<d>` in
`detailed_description`; do not repeat them in these two sections.

---

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

This budget was the systematic gap in this repo. Every ref2va prompt that
`_ref_prompt` generates without a scene still runs one shot at 42-68 words —
an order of magnitude under the range — and that is most of the shipped ref2va
set. The two scene arms added 2026-08-28 (`h3_ref2v_scene_subway`,
`h3_ref2v_scene_kitchen`) are the first that sit inside it, at four shots and
373-375 words, so the gap is now demonstrated-closable rather than universal.
Neither has been rendered, so nothing here says the budget improves output.

### 9.11 Reference-writing craft, not in the guide

All HOUSE, all worth having.

- **Treat limiting words such as "only" as strict exclusions.** "room only"
  transfers architecture, layout, surfaces, fixtures and furnishings, but not
  people, actions, wardrobe, camera, lighting, style or audio. Same shape for
  "clothes only", "face only", "body shape only", "camera only", "movement only",
  "voice only", "style only". An explicit exclusion always blocks incidental
  transfer.
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
  worth doing; naming it as a substitute for description is not.
- **Conditioning, not instruction.** The prompt is passed to the encoder
  verbatim; nothing on the other side decides whether to comply, and there is no
  negative prompt field. "Make it feel tense" conditions on the words "make it
  feel tense". "Her knuckles whiten on the railing" conditions on an image.

---

## 10. Worked examples

**These are mine, not the guides'.** They are written to this manual and are
conformant against every GUIDE rule above; where a HOUSE rule is contested I say
so rather than pretending the example settles it. The guides' own worked examples
are base §5 Cases 1-4 and ref §7.

**Every example below is GRADED, and the command is the claim.** Until
2026-09-01 this section asserted that its examples "grade clean through
`preflight_graph.py`" and nothing checked it -- preflight reads graphs, and
these are loose text. `bench/grade_prompt_text.py` closes that: it wraps a
prompt in a shipped graph of the requested mode and runs the same grader.

    python bench/grade_prompt_text.py --mode fl2va --length 345 example.txt

**A prompt is conformant AT A DURATION**, which is why every heading names one.
`S.SS` in Part One and every `At MM:SS.mmm` cut resolve against the snapped
length, so the same text is correct at 345 frames and wrong at 192. Pass
`--length` to grade at the duration the example is written for.

**For ref2va, pass `--like` a graph that wires the references the prompt
declares.** Reference labels are graded against the donor's sockets, so 10.5
reports two FAILs against a one-picture donor and none against
`h3_ref_image_audio_api`, which wires the two pictures and one audio clip it
names. The tool says so when every failure is of that shape.

### 10.1 T2VA — 243 frames, 10.125 s, two shots

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames a night ferry deck across the ten-second take, wet steel railing in the foreground and harbour lights smeared behind. A dock worker in her forties with a low, level alto (S1) leans on the rail, on-screen, unhurried delivery, and says: <d>[English] Last crossing until Thursday.</d> Her lips close and her jaw stops moving as she pushes back from the rail. A younger man in a canvas jacket stands two paces behind her and produces no vocal sound. The camera trucks right with small amplitude at slow speed, carrying the harbour lights across the frame. [Shot 2] At 00:05.000, the shot cuts to a close shot of the man's hands folding a paper timetable against his knee. The man, on-screen, mid-twenties, with a dry, slightly hoarse tenor (S2), looks up and answers: <d>[English] Then we wait it out here.</d> His lips settle closed and the paper flattens under his thumb.

overall_soundscape: Diesel engine rumble carries under a steady wash of water against the hull. Wind pulls at loose canvas, boots scuff on wet steel, and a mooring chain knocks twice against the deck plate.

non_diegetic_music: A single sustained low synth tone at a slow tempo, joined by a soft brushed-snare pulse that rises briefly and drops away before the final frame.
```

### 10.2 I2VA — five examples

#### 10.2.1 — 192 frames, 8.000 s, one shot

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the seated man shown in <Picture 1> remains at the workshop bench across the eight-second take, preserving his appearance, the grey work shirt, the bench layout, and the arrangement of tools behind him. The camera pushes in with small amplitude at slow speed as he turns the small brass mechanism a quarter turn under the lamp. Sawdust lifts in the lamp beam. The man, on-screen, sixties, with a quiet, gravelled baritone (S1), keeps his eyes on his hands and says: <d>[English] It was never the spring. It was the seat.</d> His lips close and his jaw stops moving, and he sets the mechanism down on the felt pad.

overall_soundscape: Low workshop room tone continues throughout under the hum of a bench lamp. Metal ticks against metal, felt brushes across wood, and a single drawer rolls shut near the end.

non_diegetic_music: N/A
```

#### 10.2.2 — 345 frames, 14.375 s

**No dialogue, one shot, camera opening out.** The picture's framing is held by the opening clause, and the only non-speaker on screen is given `produces no vocal sound`.

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, holding the exact framing, lighting, wardrobe and composition established in <Picture 1>, a bench joiner in a canvas apron stands over a half-planed board in a timber workshop, low sun coming through sawdust in the air behind her. She sets her weight, draws the plane the length of the board, and a single curled shaving lifts and falls to the floor. She produces no vocal sound. The camera pulls out with small amplitude at slow speed, opening the frame to the racked chisels along the back wall.

overall_soundscape: The long dry rasp of a hand plane on softwood repeats at an unhurried pace, each pass ending in the light tick of a shaving dropping to boards. Room tone is close and wooden, with a faint hum from a strip light overhead.

non_diegetic_music: N/A
```

#### 10.2.3 — 345 frames, 14.375 s

**Two shots, one speaking turn, dialogue in the second.** The cut carries the timestamp; `[Shot 1]` does not.

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, holding the exact framing, lighting, wardrobe and composition established in <Picture 1>, a woman in a grey linen shirt stands among staged seedling trays in a glasshouse, flat overcast light falling through the panes above her. She lifts one tray to eye level and turns it slowly. The camera pushes in with small amplitude at slow speed toward her hands. [Shot 2] At 00:07.000, the camera cuts to a close shot of the tray against her chest. The woman, on-screen, with a warm unhurried mezzo (S1), looks up and says: <d>[English] These two came up early.</d> Her lips close and her jaw stops moving as she sets the tray down on the bench.

overall_soundscape: A steady wash of rain on glass carries throughout, with the hollow knock of a plastic tray set on a wooden bench and the faint drip of condensation running down a pane.

non_diegetic_music: N/A
```

#### 10.2.4 — 345 frames, 14.375 s

**Documentary register, no dialogue, `non_diegetic_music` in use.** A tilt, which is one of the vocabulary's less-used types.

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, documentary, holding the exact framing, lighting, wardrobe and composition established in <Picture 1>, a fisherman in oilskins coils a wet mooring line on a harbour wall at first light, the hull of a small trawler filling the background. He works the coil twice around his forearm, drops it over a bollard, and straightens. He produces no vocal sound. The camera tilts up with small amplitude at slow speed from his hands to the masts behind him.

overall_soundscape: Water slaps against stone in an irregular rhythm under a constant low wind. Wet rope drags across concrete, a bollard takes the line with a dull knock, and gulls call intermittently at a distance.

non_diegetic_music: A sparse sustained string chord at a slow tempo, swelling gently as the frame opens upward and holding without resolution.
```

#### 10.2.5 — 345 frames, 14.375 s

**One shot, one speaking turn, mouth closed on the line.** Shows the identity-and-voice clause sitting immediately before the `<d>` block.

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, holding the exact framing, lighting, wardrobe and composition established in <Picture 1>, an archivist in a charcoal cardigan stands at a reading-room table under a green shaded lamp, a shallow box of loose photographs open in front of her. She lifts one print, angles it toward the lamp, and studies it. The archivist, on-screen, with a quiet precise alto (S1), says: <d>[English] This one was never catalogued.</d> Her lips settle closed and she lays the print face up on the table. The camera trucks left with small amplitude at slow speed along the edge of the table.

overall_soundscape: Deep room tone in a high-ceilinged reading room with a long soft reverb tail. Stiff photographic paper flexes and settles, and a chair creaks once somewhere off to the side.

non_diegetic_music: N/A
```

### 10.3 FL2VA — five examples

#### 10.3.1 — 192 frames, 8.000 s, one shot, no bracketed labels

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 8.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a kitchen table in flat morning light, beginning in the exact position and framing established by Picture 1 with a folded paper map lying closed beside a cooling mug. The camera holds a static shot through the entire eight-second duration. Two hands enter from the right, press the near edge of the map flat, and draw the first fold open; the paper lifts, creases release one at a time, and the printed coastline widens across the tabletop. The mug is nudged aside as the second fold opens, its shadow shortening as the sheet spreads over it. The map settles fully open, the hands withdraw to the frame edge, and the sheet, mug position and composition come to rest exactly as established by Picture 2 at the end of the shot.

overall_soundscape: Quiet kitchen room tone continues throughout with a faint refrigerator hum. Stiff paper crackles as each fold releases, ceramic scrapes briefly across wood, and the sheet settles with a soft rustle.

non_diegetic_music: Two alternating piano notes at a slow tempo, joined by a low sustained string that fades as the paper stops moving.
```

#### 10.3.2 — 345 frames, 14.375 s

**One shot, no dialogue.** The convergence is described as observable events -- strokes shortening, a hand taking the lane rope -- rather than as a transition into the picture.

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, one continuous shot in a tiled municipal pool hall, hard overhead light breaking on the water. A swimmer in a black cap pushes off the near wall and crosses the frame left to right in four unhurried strokes, the wake spreading behind her and flattening. She produces no vocal sound. The camera trucks right with small amplitude at slow speed, holding her shoulders in frame as she moves. Her stroke shortens, she reaches the far lane rope and takes hold of it, and her position, the settling water, the lighting and the camera's angle and framing converge on Picture 2 at the end.

overall_soundscape: Water breaks and closes over a steady four-count stroke rhythm, with a long hard reverb tail bouncing off tile throughout. A ventilation hum sits low under everything, and the lane rope rattles once as it is taken.

non_diegetic_music: N/A
```

#### 10.3.3 — 345 frames, 14.375 s

**Two shots, so Part One names `Shot 2`.** This is the case that no shipped graph exercises; see the note under section 13.

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 2) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a wide shot holds a chalkboard wall in an empty lecture room, late afternoon light raking across it from tall windows. A lecturer in a rolled-sleeve shirt works left to right, filling the board with diagrams, his back mostly to camera. He produces no vocal sound. The camera pans right with small amplitude at slow speed, following the writing as it advances. [Shot 2] At 00:08.000, the camera cuts to a medium shot from the far side of the room, the filled board now behind him. He sets the chalk in the tray, steps back once to take in the whole board, and folds his arms. His stance, the finished board, the raking light and the camera's angle and framing converge on Picture 2 at the end.

overall_soundscape: Chalk taps and drags against slate in short irregular bursts, each stroke ending with a dry click. The room carries a long empty reverb, with distant corridor footsteps passing once and fading.

non_diegetic_music: A single piano figure at a slow tempo, thinning to one sustained note as the writing stops.
```

#### 10.3.4 — 345 frames, 14.375 s

**Animation, one shot.** A style anchor in the opening clause, and a non-human subject given `produces no vocal sound`.

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Animation, glossy 3D CG, one continuous shot on a kitchen counter at night, a single pendant lamp throwing a warm pool of light onto the worktop. A short round robot with a dented copper shell and one oversized lens rolls in from frame left, stops at a spilled bag of flour, and begins sweeping it into a pile with a flat paddle arm. It produces no vocal sound. The camera pushes in with small amplitude at slow speed toward the worktop. The pile grows compact under the paddle, the robot lowers its lens toward it, and its position, the swept surface, the lamp's pool of light and the camera's angle and framing converge on Picture 2 at the end.

overall_soundscape: A fine granular sweep of powder across a hard worktop repeats in short strokes, with small servo whirs starting and stopping between them. The kitchen is otherwise quiet, with a refrigerator hum holding underneath.

non_diegetic_music: Light pizzicato strings at a moderate tempo, playful and even, settling to a single held note as the sweeping stops.
```

#### 10.3.5 — 345 frames, 14.375 s

**Two shots with one speaking turn.** Dialogue finishes in the first shot so the final frame can be still and closed-mouthed.

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 2) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames a guard in a navy uniform at a museum door, the gallery beyond her lit low for closing. She checks a wall clock, then walks the length of the doorway and puts her hand on the frame. The guard, on-screen, with a level unhurried contralto (S1), says: <d>[English] Two minutes, then we lock up.</d> Her lips close and her jaw stops moving as she turns back to the gallery. The camera trucks left with small amplitude at slow speed alongside her. [Shot 2] At 00:08.500, the camera cuts to a wide shot of the gallery from behind her, the far lights already out. She reaches the last switch, holds still a moment, and lowers her hand. Her position, the darkened gallery, the remaining doorway light and the camera's angle and framing converge on Picture 2 at the end.

overall_soundscape: Hard-soled footsteps carry across a stone gallery floor with a long cold reverb. Switches throw with a heavy mechanical clack, and the room tone drops noticeably as each bank of lights goes out.

non_diegetic_music: N/A
```

### 10.4 L2VA — five examples

#### 10.4.1 — 158 frames, 6.583 s, one shot

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 6.58-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a close shot opens on a school corridor at dusk, a girl in a green raincoat still walking with her back half to the camera and a closed locker door behind her, a plausible earlier state of the arrangement in <Picture 1>. The camera pulls out with small amplitude at slow speed as she slows, shifts the bag strap across her chest, and turns toward the lockers. The girl, on-screen, about twelve, with a light, slightly breathy voice (S1), says: <d>[English] I left it in here.</d> Her lips close and her jaw stops moving well before the end. She reaches up, presses the locker latch, and lets the door swing to the angle shown; her shoulders drop, her chin lifts, and her hand, the door angle, the corridor lighting and the exact final composition settle into <Picture 1>, mouth closed and still, as the shot ends.

overall_soundscape: Empty corridor reverb carries a distant door closing twice. Rubber soles squeak on polished floor, a bag strap slides across fabric, and a metal latch clicks once near the end.

non_diegetic_music: A sparse celeste figure at a slow tempo over one sustained low string, thinning to a single held note at the close.
```

#### 10.4.2 — 345 frames, 14.375 s

**One shot, no dialogue.** An inferred opening that arrives at the supplied last frame only at the final frame.

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, one continuous shot in a narrow record shop, afternoon light coming in through a window at the front. A man in a corduroy jacket works along a rack of sleeves, pulling one part way out, considering it, and pushing it back. He produces no vocal sound. The camera trucks right with small amplitude at slow speed, staying level with his hands as he moves down the rack. He slows, draws one sleeve fully out, and turns it to face him, and his stance, the held sleeve, the window light and the camera's angle and framing converge on the closing composition, reaching it only at the final frame.

overall_soundscape: Stiff cardboard sleeves slide and knock against each other in an irregular rhythm. The shop carries a close dry room tone, with muffled traffic through glass and the occasional creak of a floorboard.

non_diegetic_music: N/A
```

#### 10.4.3 — 345 frames, 14.375 s

**Two shots, so Part One names `[Shot 2]`.** Note the brackets: L2VA brackets both labels where FL2VA brackets neither.

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot 2]) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a wide shot holds a rooftop at dusk, city haze behind and a folded deck chair by the parapet. A woman in a long cardigan steps out through a stairwell door and crosses toward the parapet, unhurried. She produces no vocal sound. The camera pans right with small amplitude at slow speed to follow her across the roof. [Shot 2] At 00:07.500, the camera cuts to a medium shot from beside the parapet as she arrives. She unfolds the deck chair, sets it square to the view, and lowers herself into it. Her seated position, the opened chair, the fading dusk light and the camera's angle and framing converge on the closing composition, reaching it only at the final frame.

overall_soundscape: A steady rooftop wind carries throughout with occasional gusts. A metal door swings shut once behind her, the chair frame clicks as it opens, and distant traffic hums many floors below.

non_diegetic_music: A slow warm synthesizer pad, building gradually in dynamics and holding as the movement comes to rest.
```

#### 10.4.4 — 345 frames, 14.375 s

**Animation, one shot.** Shows that `<Picture 1>` is the FINAL frame and is not part of `[Shot 1]`'s opening composition.

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Animation, hand-drawn, one continuous shot in a cluttered attic study lit by a single desk lamp. A tall thin fox in a knitted waistcoat searches a stack of papers, lifting sheets aside two at a time and setting them down in a growing pile. He produces no vocal sound. The camera pushes in with small amplitude at slow speed toward the desk. His hands slow, he draws one folded sheet from near the bottom of the stack and opens it flat under the lamp, and his posture, the opened sheet, the lamp's pool of light and the camera's angle and framing converge on the closing composition, reaching it only at the final frame.

overall_soundscape: Paper rustles and slides in short overlapping strokes throughout, with the soft thump of sheets settling into a pile. The attic is close and quiet, with rain faint on a roof overhead.

non_diegetic_music: A solo clarinet line at a slow tempo, curious and unhurried, thinning to a single sustained note as the searching stops.
```

#### 10.4.5 — 345 frames, 14.375 s

**Two shots with one speaking turn.** The line lands early, leaving the close silent.

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot 2]) aligns with the 14.38-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, documentary, a medium shot frames a farrier in a leather apron at the open side of a stable yard, cold morning light flattening the scene. He works a rasp along the edge of a hoof held between his knees, steady and repetitive. The farrier, on-screen, with a low weathered baritone (S1), says: <d>[English] Nearly there, stand easy.</d> His lips close and his jaw stops moving as he returns to the rasp. The camera holds a static shot. [Shot 2] At 00:08.000, the camera cuts to a wide shot of the yard as he finishes. He lowers the hoof, straightens up, and rests one hand on the horse's shoulder. His standing position, the settled horse, the flat morning light and the camera's angle and framing converge on the closing composition, reaching it only at the final frame.

overall_soundscape: A rasp draws across horn in long even strokes with a dry grain to each pass. Hooves shift on wet cobbles, a bucket handle rings once, and the yard carries an open outdoor ambience with birdsong at a distance.

non_diegetic_music: N/A
```

### 10.5 ref2va — 243 frames, 10.125 s, two shots, two images and one audio clip

Reference chain in connection order: image (the woman), image (the workshop),
audio (a voice clip). Labels therefore resolve to `<Picture 1>`, `<Picture 2>`,
`<Audio 1>`. Both images define identity and environment only, so per ref §2.2
neither gets a standalone entry and neither is cited in `detailed_description`.

```text
subject_definitions:
<Subject 1> is the woman in <Picture 1>, in her thirties, with cropped black hair, a tan canvas apron over a grey long-sleeved shirt, and a thin leather cord at her wrist.
<Subject 2> is the bicycle-repair workshop in <Picture 2>, with a whitewashed brick wall, a pegboard of hanging tools, a bare bulb over the bench, and two frames on a floor stand.
<Audio 1> is the voice-timbre reference for <Subject 1> (S1), carrying a spoken English vocal layer at a low, even pitch.

summary:
[reference generation + audio reference] The target video shows <Subject 1> truing a wheel in <Subject 2> while a customer waits at the doorway. The two-shot exchange uses <Audio 1> as the voice-timbre reference for <Subject 1> and ends with the wheel spinning clean.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - the cropped black hair, tan canvas apron, grey long-sleeved shirt, and leather wrist cord are retained.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - the whitewashed brick wall, pegboard of hanging tools, bare bulb, and floor stand are retained.
<Audio 1>: reference - its vocal timbre and even delivery guide the speech of <Subject 1> without copying the original signal.

detailed_description:
The target video is in a warm, naturalistic documentary style with a single practical light source and a slightly muted colour palette.
[Shot 1] A medium shot establishes <Subject 2>, the bicycle-repair workshop, its whitewashed brick wall and pegboard of hanging tools lit by one bare bulb over the bench. <Subject 1> (S1), the woman with cropped black hair and a tan canvas apron over a grey long-sleeved shirt, stands at the floor stand on the right of frame with a rear wheel clamped in the truing jig, a spoke key held between two fingers of her right hand. She rotates the wheel a half turn, watches the rim pass the indicator, and stops it with the flat of her palm. The camera pushes in with small amplitude at slow speed toward the jig as the rim comes to rest. A man in a wet cycling jacket stands just inside the doorway at the left edge of frame, helmet under one arm, and produces no vocal sound. Using the low, even timbre referenced from <Audio 1>, <Subject 1> (S1) says without looking up, <d>[English] Two spokes are loose.</d> Her lips close and her jaw stops moving as she fits the spoke key onto a nipple and turns it a quarter turn.
[Shot 2] At 00:04.500, the shot cuts to a close shot over her shoulder, the rim filling the lower half of the frame and the pegboard soft behind it. Her thumb steadies the rim while the spoke key turns twice more, and the leather cord at her wrist slides against the apron edge. The camera holds a static shot as she releases the rim and sets it spinning. The man in the wet cycling jacket, on-screen at the frame edge, mid-forties, with a light, hesitant tenor (S2), steps forward and asks, <d>[English] Can I still ride it home?</d> His lips close and he shifts the helmet to his other arm. <Subject 1> (S1) watches one full rotation, then answers in the same low, even timbre referenced from <Audio 1>, <d>[English] Yes. Slowly.</d> Her lips settle closed and her jaw stops moving while the wheel keeps turning, the rim passing the indicator without touching it, and the bare bulb throws a moving band of light across the pegboard behind her.

overall_soundscape:
Quiet workshop room tone and a faint street hum continue underneath throughout. A spoke key ticks against metal in short bursts, a wheel rim hums as it spins down, and wet fabric creaks as the waiting man shifts his weight.

non_diegetic_music:
N/A
```

`detailed_description` here runs about 380 words, inside ref §5.2's 350-500 for a
generation task. That length is the point of the example: a two-sentence ref2va
body is the most common defect in this repo's shipped prompts, and a short
example would teach it.

---

## 11. Every rule, its layer, and what checks it

`bench/preflight_graph.py` reports and never refuses; `bench/check_*.py` are
red/green. "nothing" means exactly that.

**To grade a prompt that is not in a graph yet** — anything in §10, or a draft —
use `bench/grade_prompt_text.py`, which wraps the text in a shipped graph of the
requested mode and runs `preflight_graph.grade` against it. It adds no rules of
its own, so this table describes it too. It exits nonzero on FAIL only.

| rule | layer | checked by |
|---|---|---|
| base modes emit three core fields, in order | GUIDE base §2.2 | `check_prompt_guide_conformance.py`, `preflight_graph.py` |
| ref2va emits six sections, in order | GUIDE ref §1 | `check_prompt_guide_conformance.py`, `preflight_graph.py` |
| Part One is the first line, then one blank line | GUIDE base §2.1 | `preflight_graph.py` (position implied by exact-string match) |
| T2VA has no Part One | GUIDE base §2.1 | `preflight_graph.py` |
| the I2VA / FL2VA / L2VA alignment sentence is the mode's own, verbatim, with `N` and `S.SS` resolved | GUIDE base §2.1 | **`preflight_graph.py`**, exact-string against the guide's own text; verified red on a deliberately mode-swapped graph |
| FL2VA uses no angle or square brackets in Part One | GUIDE base §2.1 | `preflight_graph.py` (same exact-string case) |
| `<Picture N>` bracket convention inside the body | GUIDE base §3.1-§3.3, *shown* | `preflight_graph.py` accepts bare `Picture N` only on a two-keyframe graph |
| the prompt names exactly the labels the graph wires | HOUSE (runtime) | `check_ref_prompt_labels.py`, `preflight_graph.py`, both directions |
| label ordinals follow append-chain order; a soundtrack's `<Audio j>` precedes its own `<Video k>`; `<Audio>` is one counter | HOUSE (runtime) | `check_reference_order.py` |
| `[Shot 1]` carries no timestamp | GUIDE base §4.2, ref §5.1 | `preflight_graph.py` |
| cut times strictly increasing | GUIDE base §4.2 | `preflight_graph.py` |
| cut times fall inside the video duration | GUIDE base §4.2 | `preflight_graph.py` |
| `[Shot N] At MM:SS.mmm` format | GUIDE ref §5.1 (*stated*), base §4.2 (*shown*) | nothing — a malformed header makes preflight's shot list empty and takes three rules inert |
| the five cut phrasings; dissolve/fade/wipe on request only | GUIDE base §4.2 | nothing |
| camera motion type from the twelve-row table | GUIDE base §4.3 | **nothing** — this is the escaped instance: a shipped prompt carried `whip pan`, `tracks left` and `at medium amplitude and moderate speed` and every gate passed it |
| amplitude only `with small/large amplitude` | GUIDE base §4.3 | nothing |
| speed only `at slow/fast speed` | GUIDE base §4.3 | nothing |
| motion written as natural English inside the shot | GUIDE base §4.3 | nothing |
| style and initial composition open `[Shot 1]` (base) | GUIDE base §4.1 | nothing |
| style stated in one or two sentences **before** `[Shot 1]` (ref2va) | GUIDE ref §5.2 | nothing |
| speaker ids `(S1)`, `(S2)`, stable across shots, unconditional | GUIDE base §4.4 | nothing |
| non-vocalising characters get no id | GUIDE base §4.4 | nothing |
| compound `(S1,S2)` only for already-numbered speakers | GUIDE base §4.4 | nothing |
| identity established where the speaker first **appears** | GUIDE base §4.4 | **nothing**, and not mechanizable |
| `<d>` contains only a language tag and the verbatim words | GUIDE base §4.4 | `preflight_graph.py` checks the tag and that no marker sits inside |
| `<d>` appears only in the main field | GUIDE ref §6 (ref2va); implied base §2.2 | `check_prompt_guide_conformance.py`, `preflight_graph.py` |
| `says in an off-screen voiceover`, then lips-remain-closed | GUIDE base §4.4 | nothing |
| `<scenetrans>` in both parts of a line crossing a cut, plus a continuity phrase | GUIDE base §4.4 | nothing — **and the token matches nothing the release declares** [OPEN] |
| `<cutoff>` for speech truncated by the end | GUIDE base §4.4 | nothing — release declares `<\|cutoff\|>`; house writes the piped form [OPEN] |
| lyrics pair wraps `<d>`; caption pair is a sibling; no other nesting; pairs balanced | HOUSE | `preflight_graph.py` |
| no whitespace before `</d>`; no caption padding; markers do not start a line | HOUSE | `preflight_graph.py` (WARN) |
| no full stop before `<\|cutoff\|>` | OPEN — the stated mechanism described a pre-fix tokenizer path | `preflight_graph.py` (WARN) |
| `<\|caption_start\|>` means subtitle | OPEN — undocumented; name is the only evidence; a signage reading was withdrawn | nothing |
| on-screen text typed literally in English double quotes | GUIDE base §4.5 | nothing |
| `overall_soundscape` is 1-4 English sentences in one paragraph | GUIDE base §4.6 | nothing |
| dialogue, singing and diegetic music not repeated in `overall_soundscape` | GUIDE base §4.6 | nothing |
| `overall_soundscape` `N/A` only for requested total silence | GUIDE base §4.6 | nothing |
| `non_diegetic_music` is 1-3 sentences on instrumentation, speed, rhythm, dynamics; no mood words | GUIDE base §4.7 | nothing |
| sequenced/chronological soundscape prose | **NOT A RULE** — example-derived, retracted | n/a |
| physical action sounds excluded from the soundscape | **NOT A RULE** — §4.6 names them as belonging there | n/a |
| label-on-its-own-line layout | OPEN — base *shows* inline, ref_en's own examples disagree with each other | nothing |
| the four label types and their meanings | GUIDE ref §2 | nothing checks the semantics; presence is checked |
| a label keeps one meaning across all six sections | GUIDE ref §2 | nothing |
| identity-only image gets no standalone `<Picture N>` entry | GUIDE ref §2.2 | nothing |
| visible content from a reference video is `<Subject N>`, not `<Video N>` | GUIDE ref §2.3 | nothing |
| an `<Audio N>` bound to a speaker reuses the global `(Sx)` | GUIDE ref §2.4, §5.4 | nothing |
| video and audio ordinals are independent | GUIDE ref §2.5 | `check_reference_order.py` (runtime side) |
| a reference video does not create `<Audio N>` merely by having sound | GUIDE ref §2.5 | nothing |
| `summary` opens with a bracketed task-type prefix | GUIDE ref §3 | `check_prompt_guide_conformance.py` |
| task types come from the six-row table, joined ` + `, no repeats | GUIDE ref §3 | `check_prompt_guide_conformance.py`, parsing the guide's own table |
| `keyframe completion` only where an image is a concrete frame anchor | GUIDE ref §3 | `check_prompt_guide_conformance.py` checks the **graph** wires `MiniMaxH3AddGuide` |
| asset presence alone does not create a task type | GUIDE ref §3 | nothing |
| no new labels introduced in `summary` | GUIDE ref §3 | nothing |
| video-editing summaries open with the fixed sentence | GUIDE ref §3 | nothing |
| one `retention_analysis` line per label | GUIDE ref §4 | `preflight_graph.py` requires a retention line for every defined label |
| visual markers from §4.1's four; audio markers from §4.2's four; sets cross only on `weak_reference` | GUIDE ref §4.1, §4.2 | `check_prompt_guide_conformance.py`, `preflight_graph.py` |
| marker chosen within the role already defined | GUIDE ref §4.2 | **nothing** — a legal marker on the wrong entity is set-legal and backwards |
| added content is not a loss of fidelity | GUIDE ref §4.2 | nothing |
| no `(Sx)` in `retention_analysis` | GUIDE ref §5.4 | `preflight_graph.py` |
| labels inserted in `detailed_description` where their roles apply | GUIDE ref §5.2, §5.3 | `preflight_graph.py` WARNs on an uncited `<Subject N>` |
| `detailed_description` 350-500 words for generation tasks | GUIDE ref §5.2 | `preflight_graph.py` (WARN, base modes correctly exempt) |
| reused source dialogue: verbatim, `[unclear]`, standardised punctuation, terminal `.` `?` `!` | GUIDE ref §5.4, scoped to reuse | nothing |
| closing punctuation on **all** dialogue | OPEN — over-generalises ref §5.4 and collides with base §4.4's verbatim rule | nothing |
| frame grid `17k + 5` at 24 fps; duration honesty | HOUSE (runtime) | `h3_rules.py::snap_length`; preflight uses it for cut-time bounds |
| last cut by `duration - 2.5 s`; speech inside `0.4 s`..`duration - 0.8 s` | HOUSE / third-party | nothing |
| speech budget `2.5 x (shot_seconds - 1.0)` | OPEN — functional form assumed, one anchor, an alternative fits it exactly | nothing |
| turn cap | OPEN — house material states both a cap and no cap | nothing |
| lips close and jaw stops at the end of every line | HOUSE, from a vendor example | nothing |
| explicit "produces no vocal sound" for silent on-screen characters | HOUSE | nothing |
| L2VA closed-mouth endpoint rule | HOUSE | nothing |
| "only" as a strict exclusion; wardrobe as preserve/add/replace/modify | HOUSE | nothing |
| cast sheet: same noun phrase verbatim on every mention | HOUSE | nothing |
| name the moves that do not happen to hold a frame | HOUSE / third-party | nothing |
| whole prompt under 7,000 characters | OPEN — no source in either guide | nothing |
| supported-language list | HOUSE — no list in either guide | nothing |

---

## 12. Where the guides are silent, ambiguous, or disagree

Stated so nobody mistakes a house reading for the vendor's.

1. **`<scenetrans>` names no declared token.** base §4.4 asks for it; the release
   declares seven H3 tokens and it is not among them. A guide describing a
   mechanism whose token does not exist. Nothing here has rendered one.
2. **`<cutoff>` versus `<|cutoff|>`.** Both guides print the unpiped spelling and
   the release declares the piped one. House writes the piped form; that is a
   deliberate divergence from guide prose, not an oversight.
3. **Five of the seven declared markers appear in neither guide.**
   `<|cutoff|>`, the lyrics pair and the caption pair are undocumented. Every
   structural rule about them here is house pattern.
4. **`<|caption_start|>` has no documented meaning.** Its name and its position
   in the declared list are the only evidence.
5. **Section-label layout is example-derived and ref_en contradicts itself.**
   §7's complete example puts each label on its own line; §6's snippets are
   inline. base_en is consistently inline and states nothing. Do not enforce.
6. **Neither guide states a whole-output English rule for the base modes.** ref_en
   states it in its preamble. base_en only says "English sentences" of the two
   audio fields and "English double quotation marks" of on-screen text.
7. **Neither guide caps turns, caps characters, or lists supported languages.**
   Any such number is house or third-party.
8. **`retention_analysis` for a cited-only `<Picture N>` is unresolved by the
   text.** ref §4 says one line per reference label; ref §2 says an identity-only
   image gets no separate definition line. The guide's own §7 example resolves it
   in practice — its four `<Picture N>` and two `<Video N>` labels are cited only
   inside `<Subject N>` definitions and receive no retention lines — but that is a
   worked example, not a statement.
9. **base §2.1's L2VA template says "reference picture**s**" for a single
   picture.** Reproduce it as written anyway; it is the string the guide says the
   mode always uses.
10. **The guide never says how a shot's duration relates to a frame count**,
    because frame counts are outside its scope entirely. Every timing number in
    §3.3 and §3.4 is ours.
11. **Neither guide states a line-break rule for the main narrative field.**
    base_en's only multi-shot worked example runs `[Shot 2]` inline in one
    unbroken paragraph, and MiniMax's own reproducible t2va API payload does the
    same -- verified 2026-09-01 by counting escaped newlines inside
    `integrated_multimodal_description` in
    `coderef/MiniMax-H3/scripts/readme/reproducible-768p-t2va-request.sh`, which
    has none. Two independent artifacts, but each is the vendor's *only*
    multi-shot base-mode specimen, and both are **shown**, not stated. Our own
    shipped prompts disagree with it and with each other; §14 records the split.
    Do not enforce it.
12. **Neither guide states a limit on dialogue turns per shot.** Both of
    base_en's dialogue-bearing examples carry exactly one `<d>` block per shot,
    and that is the whole of the vendor evidence -- **shown**, at n=2, with no
    sentence behind it. `internal/PROMPTING.md` §4.3 read it as a rule ("one
    speaking turn per shot") and contradicted itself elsewhere; §14 records where
    our shipped prompts land.

## 13. Known gaps in this repo

**Three of the four bullets this section carried until 2026-09-01 had gone
false, and they are recorded here as withdrawn rather than quietly rewritten** —
a reader who remembers the old sentence needs to know it was retracted. Each was
true when written on 2026-08-28 and was overtaken within days, which is this
repo's standard failure: prose stating a fact the code already knows.

### Withdrawn 2026-09-01

Each withdrawn claim is quoted as it stood. **The replacement is a pointer, not
a new count** — that is what made these rot: they cached a fact the graphs
already know, and a corrected count would rot the same way.

- **"L2VA has no shipped prompt, and `scene_prompt()`'s L2VA branch returns
  `[Shot N]` and `S.SS` unsubstituted."** Both halves are false. For what
  L2VA actually ships, read `prompt_catalogue.md`, or ask the graphs:

      python bench/grade_prompt_text.py --list-donors

  The `scene_prompt()` defect was fixed on 2026-08-28 and that function's own
  comment records it; the function remains uncalled, so it never reached a
  graph.
- **"No shipped graph carries any marker but `<d>`."** False. The markers each
  scene carries are a column in `prompt_catalogue.md`, derived from the graphs.
- **"Every generated ref2va prompt is one shot at 42-68 words."** False as a
  universal. For the current distribution:

      python bench/preflight_graph.py workflows/*.json | grep 'the guide asks 350-500'

### Still open

- **Camera-motion vocabulary is enforced by nothing.** Re-checked 2026-09-01:
  neither `preflight_graph.py` nor `check_prompt_guide_conformance.py` tests any
  motion phrase against base §4.3's table. A denylist of terms absent from that
  table is cheap and decidable; proving every motion phrase in-vocabulary is not.
- **The ref2va word budget is preflight's only recurring WARN**, which is why it
  stopped being read as a finding. That is a reason to fix the prompts or retire
  the WARN, not to keep both.
- **No shipped graph is a multi-shot FL2VA or L2VA.** Every keyframe graph is one
  shot, so the branch that resolves `Shot N` to anything but 1 was dead until
  2026-09-01 and is exercised only by §10.3.2, §10.3.5, §10.4.3 and §10.4.5.
  That dead branch was carrying a defect: see §14.
- **Nothing regenerates `prompt_catalogue.md`.** It has a `--check` mode and no
  caller, and it had gone stale by 2026-09-01. This repo has no test runner by
  design (`docs/checks.md`), so the discipline is to run it before trusting it,
  not to wire a gate.

---

## 14. Every source that claims to govern a prompt, and where they disagree

Reconciled 2026-09-01. **Read this before citing any of them**, because they do
not carry equal weight and two of them are not authorities at all.

### 14.1 The sources, ranked by what a violation means

| # | source | on disk | standing | a violation means |
|---|---|---|---|---|
| 1 | the vendor's two guides | `internal/official_prompt_guides/` (gitignored) | **the only authority** | the prompt is off-distribution from what the model was trained on |
| 2 | the vendor's own API payloads | `coderef/MiniMax-H3/scripts/readme/*.sh` (gitignored) | evidence of practice, not a rule | you are doing something the vendor's own pipeline never emits |
| 3 | the vendor's prompt-writing skill | `coderef/MiniMax-H3/.claude/skills/h3-prompt-writing/` | **a router, not a rule set** | nothing; it states no rule of its own |
| 4 | this file | `docs/prompting.md` | our reading, layered GUIDE / OWNER / HOUSE / OPEN | depends on the layer, which every rule names |
| 5 | `internal/PROMPTING.md` | gitignored | **superseded 2026-08-28, being sunset** | nothing; cite this file instead |
| 6 | `coderef/comfyui_dagthomas/data/h3` | gitignored | **third-party, and its base guide is a FORK of the vendor's** | nothing — see §14.2b before quoting it |

### 14.2 The vendor's skill adds no rule, and its guides are byte-identical to ours

MiniMax ship a prompt-writing skill twice, at
`.claude/skills/h3-prompt-writing/` and `.agents/skills/h3-prompt-writing/`.
Checked 2026-09-01:

- **The two copies are identical** (`diff -rq`, no differences).
- **Its `references/base-en.txt` and `references/ref-en.txt` are byte-identical
  to our `internal/official_prompt_guides/` copies** — same size, same SHA-256.
  So our guide corpus is the vendor's own text, and that is now verified rather
  than assumed.
- **`SKILL.md` is 35 lines and defers entirely to those two files.** Its three
  "Output Rules" each trace to a guide statement — the English-body-with-original-
  language-dialogue rule to ref-en:5 and ref-en:217, the avoid-a-plot-summary
  rule to ref-en:7, the per-shot description checklist to ref-en:7.

**So there is no fourth authority.** A session that finds the skill and reads it
as new guidance has found a second pointer to the guides we already have.

### 14.2b The dagthomas corpus: useful, and carrying a forked guide

`coderef/comfyui_dagthomas/data/h3` is a third-party pack
(`dagthomas/comfyui_dagthomas`) with a substantial H3 prompt corpus. It is
worth reading — it independently corroborated the inline finding and supplied
the positional mouth-cue rule in §14.3. **But check what you are quoting.**

- **`guide_ref_en.md` is byte-identical to the vendor's** (SHA-256 `1e574f35…`).
- **`guide_base_en.md` is NOT. It is a FORK** — hash `3e7757fa…` against the
  vendor's `2cfebc09…`, with changed lines: passages added and vendor
  paragraphs rewritten, in the vendor's own register, unmarked as edits.
- **`guide_chain_en.md` and `guide_crossover_en.md` correspond to nothing the
  vendor ships.** The release has base-en and ref-en only.

**The trap is that hashing one guide and generalising gets you the wrong
answer**, which is a live risk here because §14.2 does exactly that check
against MiniMax's own skill bundle and it passes there. The fork's edits
contradict stated vendor rules: it rewrites base-en's "1-4 English sentences in
one continuous paragraph" for `overall_soundscape` into one sentence as a comma
list, rewrites the `non_diegetic_music` sentence budget, and adds shot-count and
word-count rules the vendor states nowhere, attributed to a thousand-prompt
measurement with no corpus behind it. Its own examples contradict two of those
added numbers.

**So use its EXAMPLES as evidence of practice, never its guide text as
authority.** Established 2026-09-01 by a peer session; the hashes re-checked here.

### 14.3 Where our shipped prompts diverged, and what happened to each

**Shot line breaks — CLOSED 2026-09-01, we conformed.** Every vendor
multi-shot specimen runs shots inline in one unbroken paragraph, and a peer
session found the same in a third corpus: `coderef/comfyui_dagthomas/data/h3`
is inline in every multi-shot specimen that follows the official three-field
grammar, and breaks lines only inside a four-section format it invented itself.
Three independent corpora, no counterexample. `LONG_T2V_PROMPT` and the four
aisle/sortline arms were collapsed to one line per field in the generator and
every carrying graph rebuilt; word counts are unchanged, only newlines.
**Consequence: clips of those scenes rendered before 2026-09-01 are not
matched-seed comparable with clips after it.** The aisle and sortline pairs
changed identically, so the description-length experiment they belong to is
intact.

**Mouth closing — WITHDRAWN 2026-09-01, we were already conformant.** This
section previously recorded our dialogue prompts as diverging on it. **That was
a bad statistic, not a finding.** The rule is POSITIONAL, not per-line: a cue
follows a dialogue line when the shot CONTINUES past it, and never when the
line ENDS the shot. Cross-tabbed that way:

| | cue | no cue |
|---|---|---|
| shot continues | ours 32, dagthomas 17, vendor 2 | ours 12, dagthomas 7, vendor 4 |
| line ends the shot | ours 0, dagthomas 0, vendor 0 | ours 1, dagthomas 16, vendor 1 |

**The provenance matters and is weaker than it looks.** The pattern is a
dagthomas observation that our corpus happens to satisfy — our mid-shot rate is
73% against their 71%. **The vendor corpus does not corroborate it**: its
decisive shot-final cell holds a single line, which is consistent with the rule
and establishes nothing, and its mid-shot cell runs the *other* way at 2 of 6.
So do not cite the positional rule as vendor practice. The only STATED vendor
rule here remains base-en:136, for voiceover, which nothing contradicts, and
§5.6's [HOUSE] label was right all along.

It is still the right STATISTIC — it is what turned an apparent 43% cue rate
into zero counterexamples and retracted the misalignment — but a statistic being
right does not make its source authoritative.

**Turns per shot — OPEN, and narrower than first stated.** A first pass here
recorded "the vendor uses exactly one `<d>` per shot in every specimen". **That
was wrong, and it was wrong by counting shot headers across a whole prompt
rather than inside the narrative field.** The vendor's reproducible ref2va
payload puts TWO dialogue turns in ONE shot; its second `[Shot 1]` string is a
cross-reference inside `retention_analysis`, not a header. So:

- **Base format:** every vendor specimen is one turn per shot, and
  `DIALOGUE_T2V_PROMPT` is not. This is the divergence
  `bench/diff_prompt_corpus.py` still reports.
- **Reference format:** the vendor is SPLIT — one turn in the guide's worked
  example, two in the payload — so it asserts nothing, and our two ref2v scene
  arms at two turns are doing exactly what the vendor payload does. They are
  **not** outliers and were withdrawn from this finding.

What survives: the two `DIALOGUE_*` prompts stack more turns in a shot than any
vendor specimen of either format. No guide states a cap (§12.12), both are
inside the speech budget, and nothing has been rendered.

**Checked and found clean:** ref-en *states* one line per item in
`subject_definitions` (ref-en:37) and `retention_analysis` (ref-en:157). Every
shipped ref2va prompt satisfies both. A first pass here flagged twelve of them by
counting labels per line, which is the wrong test — ref-en:37 explicitly allows a
source-only `<Picture N>` to be cited inside another item's line.

### 14.4 The shot regex defect this reconciliation found

`preflight_graph.grade` extracted shots with a pattern whose body group was
`[^\n]*` -- greedy to end of line. Because the guides put every shot in ONE line, `findall`
returned a single pair for a multi-shot prompt and `shots[-1][0]` was always
`"1"`. `_expected_base_alignment` then demanded `from Shot 1` — so a correct
two-shot FL2VA prompt FAILED and an incorrect one naming `Shot 1` PASSED, an
exact inversion.

It reached no shipped graph, because every shipped keyframe prompt is one shot
and the t2va path returns before reading `shots`. Found 2026-09-01 writing
§10.3.3, fixed the same day, and no shipped graph's grading changed. The
newly-live branch is red-proved: a two-shot FL2VA claiming `from Shot 1` now
fails. **This is the standing "a fix moves where a constraint applies" case —
the branch was dead, so nothing covered it, so the defect sat in it.**

### 14.5 What `internal/PROMPTING.md` still uniquely holds

It has carried a SUPERSEDED banner since 2026-08-28 and is being sunset into
this file. Its own banner names what had already rotted: §2.1's headline that
ComfyUI does not tokenize `<d>` as such (false on this install since the core
tokenizer merge), and two instructions to wire `MiniMaxH3VendorTokens`, a node
deleted 2026-08-27. **Do not follow it.**

What has moved here: §1's model facts and the Context-IR framing, §2's
presentation contract, §3's five diagnoses, §4's frame grid and speech budget
including the open two-parameter problem, §5's documented-to-work list, and §6's
user-message contract — see §15. What has NOT moved, and is the only reason to
open the file: **§7 and §7b, the LLM prompt-writer system prompts.** Those are a
tool for driving a writer model, not a statement of the rules, and
`internal/prompts/2026-08-22_{t2va,i2va,fl2va,ref2va}_system_prompt.md` are a
separately-authored set covering the same ground per mode.

---

## 15. What the model is, and what actually reaches it

Migrated from `internal/PROMPTING.md` §§1-6 on 2026-09-01 so that file can be
retired. **Everything checkable here was re-derived against source on that date
rather than copied on trust** — that file had been wrong before and carries a
SUPERSEDED banner. Each claim below says which it is.

### 15.1 What H3 is, and why the prompt carries the whole job

H3 generates **video and 32 kHz audio jointly, in one denoising pass, from one
block of text.** There is no TTS stage, no separate audio model, and no script
field: the same DiT that draws the mouth generates the voice coming out of it.
Anything you want to hear has to be in the same text that describes what you
want to see. Guidance is CFG-distilled, so **there is no negative prompt
channel** — a "do not show X" sentence conditions on the tokens of that
sentence.

**MiniMax do not intend the model to take your prompt directly.** They ship a
second hosted system, H3-Context-IR, whose only job is to turn a short idea into
the structured document H3-Base consumes, and their model card calls it critical
to output quality. It is API-only. **Running H3 locally means you are the
Context-IR**, and that is why the guides read like an output spec rather than
like advice — they document Context-IR's output format.

**Do not hand a writer model the name "H3-Context-IR".** H3 postdates the
training cutoff of any model you would run this on, so the name retrieves
nothing and invites confabulation about what that system does; those inventions
then compete with the rules you actually wrote. State the properties instead.
The same failure in the other direction is a writer reaching for Sora or Veo
conventions to fill the gap. *(Reasoning, not measurement — carried forward from
`internal/PROMPTING.md` §1, which records it as a correction to its own earlier
draft.)*

### 15.2 What ComfyUI actually sends the encoder

**Verified 2026-09-01 against `comfy/text_encoders/minimax.py`.**

- Conditioning is the **unnormalized hidden state after LM layer 50**; the
  converted checkpoint is truncated there (`comfy/text_encoders/minimax.py:15`).
- The presentation is **not chat-templated** — raw token ids, no system or user
  roles, vision blocks spliced inline (`comfy/text_encoders/minimax.py:3`).
- **Your prompt is passed through verbatim.** `tokenize_with_weights` ends with
  `add_text(text)` (`comfy/text_encoders/minimax.py:197`) after every label; nothing is stripped,
  parsed, reformatted, or reordered.
- **Nothing injects a duration, an alignment line, or shot scaffolding.** What
  ComfyUI prepends is only the `"<Picture i>: "` / `"<Video k>: "` /
  `"<Audio j>: "` label per reference, plus a `"<%.1f seconds>"` marker before
  every 2-frame temporal block of a reference video. Everything else is yours,
  **and the writer is blind to clip length unless you tell it.**

**Consequence: the prompt is conditioning, not instruction.** There is no
assistant deciding whether to comply. Descriptive text naming what is on screen
conditions the generation; an imperative conditions on the tokens of the
imperative.

**Reference ordinals follow item order, and a video's soundtrack jumps the
queue.** Counters are per kind and independent, so the same file can be
`<Video 1>` and `<Audio 2>`. A reference video's soundtrack takes its
`<Audio j>` label **before its own `<Video k>`**, and the standalone audio loop
runs after — so one soundtracked video plus one standalone clip gives
`<Audio 1>` = the soundtrack. Verified in both
`comfy_extras/nodes_minimax_h3.py:335,346` and this pack's
`reference_conditioning.py:655-696`.

### 15.3 The five ways prompts go wrong here

Carried forward from `internal/PROMPTING.md` §3. These are diagnoses from
observed outputs, not measurements.

1. **The writer does not know what H3 is**, so it writes for a generic
   text-to-video model: it hedges, writes instructions rather than descriptions,
   and has no reason to treat the audio fields as load-bearing. §15.1 is the fix.
2. **Named characters come back as generic archetypes.** H3 renders what is
   *described*, not what is *named* — the DiT was trained on caption-shaped
   descriptions of pixels, so a bare proper noun gives it nothing to draw. The
   fix is a **cast sheet pass**: expand every named character, IP or real person
   once into a canonical visual noun phrase plus a voice line, then reuse that
   exact phrase verbatim on every mention. Keep the name too; never rely on it.
   Naming the *property* as a style anchor is worth doing.
3. **Nothing in the prompt knows how long the clip is** (§15.2), so a writer
   picks cut timestamps by feel and overruns the clip. §3.3 is the fix.
4. **Line breaking has no single convention** and the vendor's own artifacts use
   two — see §12.11 and §14.3, which is where this now lives.
5. **Characters talk over each other**, from three causes: two speakers in one
   shot with no time anchor (a timestamped cut is the only hard temporal anchor
   the format offers); a **compound speaker id used by mistake** — `(S4,S5)` is
   the documented notation for literal simultaneous group speech, so it
   instructs two characters to say the line in unison; and simply too many
   speakers for the runtime. Every on-screen character not given an explicit
   "produces no vocal sound" is a candidate for the model to voice anyway.

### 15.4 The speech budget, and why its shape is unsettled

`internal/PROMPTING.md` §4.2 proposed `max_words_in_shot = 2.5 x (shot_seconds
- 1.0)`, anchored on MiniMax's own ref2va script: 5 seconds, one shot, 11 spoken
words. **That anchor does not fit the formula** — it allows 10 — and the formula
has two free parameters where one observation can constrain at most one. A
one-parameter form fits the same point exactly:

| shot | `2.5 x (s - 1.0)` | `2.2 x s` |
|---|---|---|
| 2 s | 2.5 | 4.4 |
| 3 s | 5.0 | 6.6 |
| 5 s | 10.0 | **11.0 (exact)** |
| 10 s | 22.5 | 22.0 |

Both are equally "validated" by the corpus and they diverge by up to 1.8x
exactly where the decisions are tightest. **The `-1.0` intercept is doing all
the work in the short-shot regime, where there is no data at all** — and the
familiar "cutting costs you words" tradeoff is *generated by* that intercept.
Under `2.2 x s` cutting is free and the fewer-longer-shots argument disappears.

**So treat every short-shot budget as possibly 1.8x too tight, and do not cite
either form as measured.** [OPEN] Weak corroboration only: an independently
built third-party kit of 26 worked examples has a densest dialogue density of
0.71 words/sec, roughly a fifth of this budget, with no two-speaker example
anywhere in it. That is revealed preference from someone staying far under any
plausible ceiling; it supports the direction and says nothing about the constant.

### 15.5 The user-message contract for a writer model

If you drive an LLM to write these, the user turn needs more than an idea:

```
idea:     <one or two lines>
frames:   243            # prefer frames; seconds alone cannot be snapped honestly
task:     t2v            # t2v | i2v | fl2v | l2v | ref2v
```

**Give frames, not seconds.** "10 seconds" does not determine a frame count —
240 snaps to 243 (10.125 s) but 250 snaps to 260 (10.833 s) — so a writer handed
seconds cannot honestly produce the `S.SS` a Part One line needs (§3.3).

For ref2va the turn also needs the reference list **in the order the items are
built**, because that is what fixes the ordinals, including the soundtrack rule
in §15.2:

```
refs:  1 image (identity), 1 video with sound (camera motion), 1 audio (voice)
       -> <Picture 1>, <Audio 1> = video soundtrack, <Video 1>, <Audio 2> = voice
```

Aspect ratio is a workflow setting, not a prompt fact; leave it out.

### 15.6 Third-party observations, held at arm's length

None of these is measured here. They are recorded because they are actionable
and cheap to test, and because a reader who meets them elsewhere should know
this repo has neither confirmed nor refuted them. [3rd]

- **H3 defaults to moving the camera.** Say nothing and expect a slow drift; the
  reported fix is naming the moves it should *not* make rather than asking for a
  static shot.
- **Negative constraints are contested.** One vendor-adjacent guide calls
  negative lists where "most of the quality lives"; another ran matched A/B
  pairs and could not show they did anything, narrowing their advice to
  on-screen text and camera movement only — the two things the model volunteers
  unprompted.
- **One primary change per beat**, or two changes collapse into whichever is
  easier to render.
- **Wardrobe drifts even when faces hold**, so name the garment in text as well
  as showing it in a reference.
- **Quiet scenes come back genuinely quiet** and may need gain in post.

