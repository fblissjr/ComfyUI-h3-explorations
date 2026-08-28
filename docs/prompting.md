# How to write an H3 prompt

last updated: 2026-08-28

A working manual. Everything needed to write a conformant prompt for any mode is
restated here: the closed vocabularies in full, the exact Part One templates, the
section layouts, and one worked example per mode. You do not need `internal/` to
use this file, and you should not need any other file to write a prompt.

Companions: [`prompt_catalogue.md`](prompt_catalogue.md) is what we currently
render (generated); [`prompt_audit.md`](prompt_audit.md) is whether those follow
these rules. Neither is a source for the rules themselves.

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

This budget is the systematic gap in this repo: every generated ref2va prompt
currently runs one shot at 42-68 words.

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

**These five are mine, not the guides'.** They are written to this manual and are
conformant against every GUIDE rule above; where a HOUSE rule is contested I say
so rather than pretending the example settles it. The guides' own worked examples
are base §5 Cases 1-4 and ref §7.

### 10.1 T2VA — 243 frames, 10.125 s, two shots

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames a night ferry deck across the ten-second take, wet steel railing in the foreground and harbour lights smeared behind. A dock worker in her forties with a low, level alto (S1) leans on the rail, on-screen, unhurried delivery, and says: <d>[English] Last crossing until Thursday.</d> Her lips close and her jaw stops moving as she pushes back from the rail. A younger man in a canvas jacket stands two paces behind her and produces no vocal sound. The camera trucks right with small amplitude at slow speed, carrying the harbour lights across the frame. [Shot 2] At 00:05.000, the shot cuts to a close shot of the man's hands folding a paper timetable against his knee. The man, on-screen, mid-twenties, with a dry, slightly hoarse tenor (S2), looks up and answers: <d>[English] Then we wait it out here.</d> His lips settle closed and the paper flattens under his thumb.

overall_soundscape: Diesel engine rumble carries under a steady wash of water against the hull. Wind pulls at loose canvas, boots scuff on wet steel, and a mooring chain knocks twice against the deck plate.

non_diegetic_music: A single sustained low synth tone at a slow tempo, joined by a soft brushed-snare pulse that rises briefly and drops away before the final frame.
```

### 10.2 I2VA — 192 frames, 8.000 s, one shot

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the seated man shown in <Picture 1> remains at the workshop bench across the eight-second take, preserving his appearance, the grey work shirt, the bench layout, and the arrangement of tools behind him. The camera pushes in with small amplitude at slow speed as he turns the small brass mechanism a quarter turn under the lamp. Sawdust lifts in the lamp beam. The man, on-screen, sixties, with a quiet, gravelled baritone (S1), keeps his eyes on his hands and says: <d>[English] It was never the spring. It was the seat.</d> His lips close and his jaw stops moving, and he sets the mechanism down on the felt pad.

overall_soundscape: Low workshop room tone continues throughout under the hum of a bench lamp. Metal ticks against metal, felt brushes across wood, and a single drawer rolls shut near the end.

non_diegetic_music: N/A
```

### 10.3 FL2VA — 192 frames, 8.000 s, one shot, no bracketed labels

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 8.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a kitchen table in flat morning light, beginning in the exact position and framing established by Picture 1 with a folded paper map lying closed beside a cooling mug. The camera holds a static shot through the entire eight-second duration. Two hands enter from the right, press the near edge of the map flat, and draw the first fold open; the paper lifts, creases release one at a time, and the printed coastline widens across the tabletop. The mug is nudged aside as the second fold opens, its shadow shortening as the sheet spreads over it. The map settles fully open, the hands withdraw to the frame edge, and the sheet, mug position and composition come to rest exactly as established by Picture 2 at the end of the shot.

overall_soundscape: Quiet kitchen room tone continues throughout with a faint refrigerator hum. Stiff paper crackles as each fold releases, ceramic scrapes briefly across wood, and the sheet settles with a soft rustle.

non_diegetic_music: Two alternating piano notes at a slow tempo, joined by a low sustained string that fades as the paper stops moving.
```

### 10.4 L2VA — 158 frames, 6.583 s, one shot

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the 6.58-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a close shot opens on a school corridor at dusk, a girl in a green raincoat still walking with her back half to the camera and a closed locker door behind her, a plausible earlier state of the arrangement in <Picture 1>. The camera pulls out with small amplitude at slow speed as she slows, shifts the bag strap across her chest, and turns toward the lockers. The girl, on-screen, about twelve, with a light, slightly breathy voice (S1), says: <d>[English] I left it in here.</d> Her lips close and her jaw stops moving well before the end. She reaches up, presses the locker latch, and lets the door swing to the angle shown; her shoulders drop, her chin lifts, and her hand, the door angle, the corridor lighting and the exact final composition settle into <Picture 1>, mouth closed and still, as the shot ends.

overall_soundscape: Empty corridor reverb carries a distant door closing twice. Rubber soles squeak on polished floor, a bag strap slides across fabric, and a metal latch clicks once near the end.

non_diegetic_music: A sparse celeste figure at a slow tempo over one sustained low string, thinning to a single held note at the close.
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

## 13. Known gaps in this repo, as of this file's date

- **L2VA has no shipped prompt**, and `workflows/build_workflows.py`'s
  `scene_prompt()` L2VA branch returns `[Shot N]` and `S.SS` unsubstituted where
  `fl2v_prompt()` resolves both. The missing scene hides a live bug.
- **No shipped graph carries any marker but `<d>`.** The lyrics pair, the caption
  pair and `<|cutoff|>` live only in generator scenes reachable through
  `--print-scene`.
- **Camera-motion vocabulary is enforced by nothing**, and it now has its escaped
  instance. A denylist of terms absent from base §4.3's table is cheap and
  decidable; proving every motion phrase in-vocabulary is not.
- **Every generated ref2va prompt is one shot at 42-68 words** against ref §5.2's
  350-500. This is preflight's only recurring WARN, which is why it stopped being
  read as a finding.
