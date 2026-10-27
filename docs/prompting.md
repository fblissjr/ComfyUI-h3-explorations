# How to write an H3 prompt

last updated: 2026-10-11 (one row in "Where everything else lives", pointing at the masked lane's page); 2026-10-01

**The single source of truth for writing an H3 prompt, in any mode.** Everything
needed is restated here: the closed vocabularies in full, the exact Part One
templates, the section layouts, all seven markers, graded worked examples
per mode, what the model actually receives, what to pin and what to leave to
the model (§16), and where every source that claims to govern a prompt
disagrees with the others. **You do not need `internal/` and you should not
need any other file for a rule.**

`internal/PROMPTING.md` was superseded and is now DELETED (2026-09-01); §14.5
says where its content went. §14 is the reconciliation across the sources that
claim to govern a prompt — read it before citing any of them, because two are
not authorities.

**Writing one now?** Read in this order: §1 the mode, §2 its structure, §3
shots and timing, §4 to §8 camera, speakers, text, markers and the audio
fields, §9 if it has references, §16 what to pin. Grade the draft with
`bench/grade_prompt_text.py` (§10).

## Where everything else lives

This file owns the rules and the rest of the repo points here, so a rule is
edited in one place. Where another document restates a rule and disagrees,
this file is right and the other is stale.

| question | go to |
|---|---|
| Where does prompt text live, and where do I start from? | [`prompt_bank/`](../prompt_bank/), one file per prompt; every prompt a graph renders is there. [`prompt_bank.md`](prompt_bank.md) is its generated table: `ships` says which graphs render an entry, `adapt` is mechanical and not a permission to render at another length, `tests` says what a scene is a test of. An experiment names a prompt by bank id: `bench/run_graph_arms.py` takes `@bank:<id>`, and `workflows/prompts.py::describe` is what every render row records |
| What does the repo render? | [`prompt_catalogue.md`](prompt_catalogue.md), generated from the graphs |
| Is a shipped prompt any good? | [`prompt_audit.md`](prompt_audit.md), one hand-written verdict per scene |
| Reference sizing and the runtime label order | [`h3_references.md`](h3_references.md) |
| Legal canvases and lengths | [`h3_resolutions.md`](h3_resolutions.md), [`h3_geometry_and_nodes.md`](h3_geometry_and_nodes.md) |
| Comparing two prompts fairly | [`eval_comparison.md`](eval_comparison.md). The one measured length pair is [`prompt_length_experiment.md`](prompt_length_experiment.md) |
| Why a rule was added, reversed or withdrawn | [`wiki/decisions.md`](wiki/decisions.md) |
| Handing the rules to someone outside the repo | [`portable/h3_prompt_standard.html`](portable/h3_prompt_standard.html), and a writer-model version at [`portable/h3_system_prompt.md`](portable/h3_system_prompt.md). Both are copies of this file; `portable/snapshots.json` pins the frozen dated ones, and a copy published elsewhere is a snapshot of whatever it was published from |
| An agent routing by task | `.claude/skills/h3-prompt` |
| The text of a masked video-to-video load, and what it says of a motion video | [`wiki/masked_v2v.md`](wiki/masked_v2v.md): its section on the Masked Prompt node for what that node writes, and "The motion video as a body model, with marks drawn on its face" for the lines that say what such a video is and that the subject follows it. **That second one is provisional and ours (HOUSE, 2026-10-11): it held on one shot and did not repeat on a second.** It is a practice written down with its evidence, not a rule of this file |


## Sections 5.8–16 (Modular Leaf Specs in `docs/prompting/`)

Sections `1`–`5.7` (mode selection, per-mode structure and Part One templates, shots and timing, camera motion vocabulary, and speaker IDs / dialogue / mouth-closing / `ref2va` dialogue) are in this file below. Sections `5.8`–`16` live in modular `< 30 KB` leaf files under `docs/prompting/`:

| Leaf file | Sections & scope |
|---|---|
| [`docs/prompting/06_singing_text_markers_and_audio.md`](prompting/06_singing_text_markers_and_audio.md) | Singing, Dialogue Fit, On-Screen Text, Markers & Audio Fields (§5.8–§8) |
| [`docs/prompting/09_references_ref2va.md`](prompting/09_references_ref2va.md) | References (`ref2va` Only: Labels, Numbering, Summary & Retention Analysis, §9) |
| [`docs/prompting/10_worked_examples.md`](prompting/10_worked_examples.md) | Graded Worked Examples (`T2VA`, `I2VA`, `FL2VA`, `L2VA`, `ref2va`, §10) |
| [`docs/prompting/11_rule_matrix_and_gaps.md`](prompting/11_rule_matrix_and_gaps.md) | Rule Matrix, Guide Disagreements & Known Gaps (§11–§13) |
| [`docs/prompting/14_sources_and_internals.md`](prompting/14_sources_and_internals.md) | Governing Sources, Encoder Internals & Failure Modes (§14–§15) |
| [`docs/prompting/16_specificity.md`](prompting/16_specificity.md) | Specificity: What to Pin and What to Leave to the Model (§16) |

## Four layers, and every rule says which one it is

Collapsing these is how two invented rules shipped here and were retracted.

| layer | what it is | breaking it means |
|---|---|---|
| **GUIDE** | the vendor's own text, `vendor_guides/base_en.md` and `ref_en.md`, cited by section | the prompt is **off-distribution** from what the model was trained on |
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

ref §5.2's opening example, verbatim (the vendor's timestamped header; the house writes `[Shot 2] The shot cuts to ...`, section 3.1):

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

> **OWNER RULE since 2026-09-18: shot headers carry NO timestamps.**
> Write `[Shot 2] The shot cuts to ...`, never `[Shot 2] At 00:05.200, the shot
> cuts to ...`. "The only time you should use timestamps in prompts is if you
> break stuff up mid-shot", that is, a time that splits action INSIDE one shot,
> not one that opens a shot. This departs from both vendor guides, whose stated
> format is quoted below and kept for reference; the owner's judgement is that
> fixed cut times are "probably a big cause of issues" on a model that is never
> told the clip's duration. Every header timestamp was removed from
> `prompt_bank/` on that date. **Measured once, and the one pair slightly
> favoured the guides.** One same-seed pair of the diner prompt, with and
> without its cut times, unblinded (`bench/results/2026-09-18_timestamps_diner.md`):
> - both versions cut twice at nearly the same points;
> - the written times were not hit exactly either;
> - the owner judged the timestamped take "a better scene", with "nothing really
>   wrong with either".
>
> The rule stands on the owner's judgement, not on that pair. Until
> 2026-09-26 this paragraph said the rule was "NOT yet measured in either
> direction" (`docs/wiki/decisions.md`). The graded examples of section 10 and
> the rule table were converted with the bank. `preflight_graph.py` now FAILS
> any shot header that opens with a time; its older cut-time rules (strictly
> increasing, inside the clip's duration) still run on whatever times a prompt
> has, so they now guard the mid-shot exception.
>
> **The one place a time belongs, in the owner's words (2026-09-26):**
> "timestamps are ONLY to be used when you are changing shots mid-shot... you
> start shot 2, and then 2 seconds later, you want it to cut to X. That's a
> timestamp. But if you're just cutting from shot 1 to 2 to 3, no timestamps
> should be in the prompt."
>
> - **A cut to the next numbered shot takes no time:** `[Shot 3] The shot cuts
>   to ...`.
> - **A change inside a shot, before the next `[Shot N]` header, takes one:**
>   `[Shot 2] The shot cuts to a medium shot of the agent on the landing. ...
>   At 00:07.500, the camera cuts to a close-up of his hand on the rail. ...`
>   The time says where inside shot 2 the change lands. It is measured from the
>   clip's start, and it must fall after shot 2 opens and inside the clip's
>   duration.

base §4.2 and ref §5.1, *stated* (the vendor format; superseded for shot
headers by the house rule above):

- **`[Shot 1]` carries no timestamp.** Do not add one.
- Later shots use sequential numbers and open with a **strictly increasing** cut
  time that **falls within the video duration**.
- The format is `[Shot N] At MM:SS.mmm, ...` (spelled out in ref §5.1; base §4.2
  shows it: `[Shot 2] At 00:03.500, the camera cuts to...`).

**FL2VA defaults to one shot** (base §3.2, *stated*): it "generally favors a
single shot so the model can interpolate continuously from the first frame to
the last frame", and uses multiple shots only when they are explicitly
specified. When it does cut, the last frame is reached by the final `[Shot N]`
at the end of the video.

A malformed header such as `[Shot 1, 00:00.000-00:06.000]` is not merely
non-conformant: the shot rules need a literal `]` after the digits, so the shot
list used to come back empty and **three shot rules went inert silently**. Since
2026-10-01 `bench/preflight_graph.py` FAILs it (`header_problems`) instead of
dropping the shot.

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

### 3.5 Landing on a keyframe: I2VA, L2VA, FL2VA

All HOUSE, from an externally written writer prompt. The guides give the
direction (base §1 and §3.3: converge from a plausible earlier state onto the
frame) but not these.

- **I2VA starts from the picture and moves forward.** `<Picture 1>` is the
  exact frame at 0.00 s, not loose inspiration and not the final frame. Do not
  return to the opening composition unless the brief asks for it.
- **L2VA and FL2VA plan backward from the final frame.** Work out the earlier
  state the action needs, then converge on the picture's subjects, pose,
  expression, clothing, objects, lighting, focus and framing, and reach it only
  at the exact end: not early, not held for time nobody asked for, and nothing
  after it. Leave time for the last movements to settle; §5.6 has the
  closed-mouth case.
- **Describe the events, not the transition.** "The video transitions into the
  final picture" gives the model nothing to render. Name the physical,
  environmental or camera events that produce the final state, and invent no
  transformation that ordinary action or camera movement already covers.
- **Nothing crosses between endpoints unexplained.** A subject, object or
  detail visibly absent from one endpoint does not appear in it; bring it in or
  take it out through a visible event in between. FL2VA also does not repeat
  either endpoint or drift back to the opening composition.

---

## 4. Camera motion: the closed vocabulary

base §4.3, *stated*. A complete expression has three dimensions and **all three
draw from closed sets**. Anything outside them is something the guide does not
attest.

**OWNER ruling, 2026-09-01: camera wording that aligns with common
cinematography is acceptable.** The table is the attested vocabulary, not a
closed set to police, and a `dolly` on a shot built around a rail dolly is the
honest description of it. So a motion type outside the table is not a defect
by itself: `bench/check_camera_vocabulary.py` warns on one and never fails on
one, the amplitude and speed phrases are the part it fails on, and a warned
term is not fixed without a rendering reason. The table is still what keeps a
prompt in-distribution, which is why the warning stays. Revisit only if output
quality gives a reason. [OWNER]

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
- **A `[Shot N]` header already IS the cut.** `whip pan` is
  absent from the table entirely, and in the prompt where it shipped it also
  re-described a cut the header had already made. Write the cut, then the move.

Two more, house-level and useful: **H3 reframes by default**, so to hold a frame
name the moves that do not happen as well as asking for a static shot; and
**one primary change per beat**, because two changes collapse into whichever is
easier to render. [HOUSE, third-party]

Three more, from an externally written writer prompt [HOUSE]:

- **One coherent camera path per shot.** Phases can follow one another inside a
  continuous shot if you say how one becomes the next, but a camera cannot be
  static while it tracks, pans, zooms, shakes or reframes.
- **Unambiguous framing.** "Medium two-shot, framed from the waist up"; "full
  two-shot, both subjects head to feet". Never combine "medium" and
  "full-body".
- **In `POV`, looking up, down or sideways is camera rotation, not a cut**, and
  the shot shows only what the subject can see.

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

**In ref2va the id is reused at every vocal event, and ref §5.4 states it.**

> Assign `(Sx)` once according to the order of actual vocal events in the target
> video. Reuse the corresponding ID at every actual vocal event in
> `detailed_description`.

The same paragraph states the one exemption, and states it as a prohibition:
verbal content that is "only a cue within a directly reused BGM or complete
soundtrack", produced by no person, character, narrator or other independent
vocal source, takes `<Audio N>` as its audible source and you **do not invent an
additional `(Sx)`**. A voice a concrete source physically produces takes one.

**base_en states neither.** Its §4.4 requires a vocalising subject to *have* a
stable id and to keep it across shots, and says nothing about repeating that id
at each vocal event; its two worked examples carry one per line, but an example
is not a rule. So the reuse rule binds ref2va and the base modes are silent,
which is the guides' asymmetry and not ours (§12.13).
`bench/preflight_graph.py::speaker_id_rules` FAILs the ref2va case and reports the
base one. [GUIDE ref §5.4, *stated*]

### 5.2 Where identity goes

base §4.4, *stated*: **when a speaker first appears**, provide enough information
from the visual and audio context to establish a stable identity — character
type, age, gender, whether the person is on-screen, pitch, timbre, speaking rate,
accent.

Note "first **appears**", not "first speaks". Introducing `(S2)` in one shot and
describing them in the next is a violation, and it is a live defect in a shipped
house prompt.

**A speaker is placed in the frame, and base §4.1 states it** in the sentence
that governs the whole description: every detail corresponds to something
visible or audible, among them "subject appearance **and position**". ref §5.3
states it again for a referenced subject — describe "its referenced
characteristics, **position in the frame**, and current action". The failure is
specific and has rendered: a speaking character with no stated position, in a
shot where a silent character has one, gets staged out of frame and the model
gives the line to whoever it did place. Saying where each speaker is costs a
clause. [GUIDE base §4.1 and ref §5.3, *stated*; the render is
`bench/results/2026-09-18_sol_reorder_panel.md`, the cafe pair]

**The voice has to fit the identity the same sentence establishes.** §4.4's list
names "character type, age, gender ... pitch, timbre" together as what makes the
identity *stable*, so a descriptor pulling against another element of it is not
a stable identity: four women in the bank were given a baritone. Which way to
resolve it is ours, and the resolution is to move the range, not the character.
[GUIDE base §4.4 for the stable-identity requirement; HOUSE that a baritone
contradicts the rest of the description]

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
material: at the moment a line ends, describe the lips closing and the jaw
ceasing to move, or the mouth keeps moving past the audio. [HOUSE, from a
vendor example]

**The pattern is positional.** Write the cue after a line the shot continues
past, and none after a line that is the last thing in its shot, because the cut
ends it; the one exception is the L2VA endpoint rule below. §14.3 has the
cross-tab and says how weak its provenance is: a third-party corpus pattern
that ours happens to satisfy, not vendor practice. The worked examples in §10
write the cue at a shot's end as well; they are conformant, because nothing
enforces either reading. [HOUSE]

Companions, all HOUSE:

- Every on-screen character who **never vocalises in the clip** gets an explicit
  "produces no vocal sound", or the model may voice them. Someone silent only
  for now is written as what they do instead ("does not answer"). §15.3 item 5
  owns the rule and the reason.
- **L2VA endpoint rule**: if the final picture shows a closed-mouth expression,
  finish the dialogue early enough for the mouth to return to that expression
  before the endpoint. A closed-mouth expression cannot remain physically
  unchanged during speech.
- **When speakers alternate, finish the first speaker's line and mouth movement
  before the next begins.** [external writer prompt]

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


---

## Sections 5.8–16

Moved to modular leaf files under `docs/prompting/` so every file fits in a single read call (`< 40 KB`):
- [`docs/prompting/06_singing_text_markers_and_audio.md`](prompting/06_singing_text_markers_and_audio.md) — Singing, Dialogue Fit, On-Screen Text, Markers & Audio Fields (§5.8–§8)
- [`docs/prompting/09_references_ref2va.md`](prompting/09_references_ref2va.md) — References (`ref2va` Only: Labels, Numbering, Summary & Retention Analysis, §9)
- [`docs/prompting/10_worked_examples.md`](prompting/10_worked_examples.md) — Graded Worked Examples (`T2VA`, `I2VA`, `FL2VA`, `L2VA`, `ref2va`, §10)
- [`docs/prompting/11_rule_matrix_and_gaps.md`](prompting/11_rule_matrix_and_gaps.md) — Rule Matrix, Guide Disagreements & Known Gaps (§11–§13)
- [`docs/prompting/14_sources_and_internals.md`](prompting/14_sources_and_internals.md) — Governing Sources, Encoder Internals & Failure Modes (§14–§15)
- [`docs/prompting/16_specificity.md`](prompting/16_specificity.md) — Specificity: What to Pin and What to Leave to the Model (§16)
