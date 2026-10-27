> **Part of [`docs/prompting.md`](../prompting.md) (Singing, Dialogue Fit, On-Screen Text, Markers & Audio Fields (§5.8–§8)).**
> For the core prompt structure (`§1`–`§5.7`) and master router, see [`docs/prompting.md`](../prompting.md).

### 5.8 Singing

**Added 2026-09-01. This section was missing while both derived extracts
carried the rule**, so the portable standard was ahead of this file and its
footer's claim that the manual is newer was, for a day, false. The underlying
fact was here as a language-preservation clause in §2 and never as guidance for
writing a sung line.

**Lyrics go in the dialogue block. There is no separate lyrics tag in either
guide** — ref §5.1, *stated*: "Give vocal sources stable `(S1)`, `(S2)`, and
subsequent IDs. Write dialogue and lyrics as `<d>[Language] ...</d>`."

```text
The busker with a low, unhurried contralto (S1) sings: <d>[English] Wait for the tide to turn.</d>
```

- **A singer is a vocal source**, so takes a stable id exactly as a speaker
  does, keeps it across shots, and a character who never vocalises gets none
  (§5.1). Two singing together take `(S1,S2)`, which means in unison.
- **Preserve the original words and language verbatim inside `<d>`** — the same
  rule as spoken dialogue, and the reason the language tag is there.
- **Singing does not go in `overall_soundscape`** (base §4.6, *stated*: that
  field is ambience, physical action and *non-verbal* human sound). **Diegetic
  singing — anything the characters can hear — belongs in the description, not
  in `non_diegetic_music`** (base §4.7, *stated*).

**The declared lyrics markers are a HOUSE pattern under test, not a rule and
not a prohibition.** The release declares `<|lyrics_start|>` and
`<|lyrics_end|>`; neither guide mentions them, so the guides' only stated way to
write a sung line is the `<d>` block above. §7 defines how this repo wraps a
sung run with them, and the two `h3_ref2v_scene_*` marker arms emit them
deliberately — they are the reason those graphs exist. **Nothing has been
rendered and judged**, so treat the pair as OPEN: following the guide alone
means `<d>` and nothing more, and using the markers is us testing whether the
declared tokens do anything. Do not read §7's structure as a licence, and do not
read this paragraph as a ban.

### 5.9 Who a line is spoken to

**Added 2026-09-01, the same lag §5.8 records for singing:** both derived
extracts carried this rule while this file had only the quoted ref §5.4 example
in §5.7 and no rule sentence, so for a day the portable standard was ahead of
the file it says it derives from.

When more than one person is present, say who the line is spoken **to**. The
addressee goes in the action outside `<d>`, named by what is visible — "turns
toward the woman in the charcoal coat" — or by its subject label in ref2va. ref
§5.4's own instance:

```text
<Subject 2> (S1) turns toward the woman and says, <d>[English] Last summer, I went to my grandfather's house. He talked about you.</d>
```

- **The slot is GUIDE.** base §4.4, *stated*: the identifying phrase, id,
  action and delivery go outside `<d>`, and an addressing action is an action.
- **Naming the addressee is GUIDE-*shown*, not stated.** ref §5.4 shows it once
  and neither guide states it as a rule; a joined-line search of both guides
  for "spoken to", "addressee" and "listener" finds nothing. Doing it whenever
  a second person is present is HOUSE.
- **A listener never takes a speaker id.** base §4.4, *stated*: characters who
  never vocalise receive no id. Ids belong to voices, so an id on someone who
  is only listening declares a vocal source the clip then has to fill. The
  example above gives the woman no id because she does not speak.
- How reliably a model follows an addressing cue is **unmeasured** here; the
  guides show it once. Write it, and do not assume it lands. [OPEN]

### 5.10 Fit the dialogue to the shot, and the shots to the scene

**Too many words in a shot make the speaker rush through them**, an unnatural
fast delivery; too few in a long shot leave dead air unless action fills it.
The owner's observation on renders, 2026-09-11. [HOUSE, observed, not
measured]

- **Budget from speaking time, not shot length.** Take off the action before
  the line and the mouth closing after it (§5.6), then fit the words into what
  is left. §3.4 carries the working budget and §15.4 why its shape is
  unsettled. The rates in circulation sit between about two and 2.8 spoken
  words per second and none is measured; an external writer prompt uses about
  two per second of actual speaking time, and fewer for stylized or emotional
  delivery, several speakers, or action during the line. When in doubt, cut
  words. [HOUSE]
- **Shorten generated dialogue until it fits; never pad it** with filler,
  babble or vocal lead-ins outside `<d>`. [HOUSE, external writer prompt]
- **Cuts and shot counts come from the scene's beats**, not from an earlier
  prompt: copying one prompt's shot count or pacing into another is the
  failure this section was written against. The last-cut margin is in §3.4.
  [HOUSE]

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
| `<\|cutoff\|>` | speech truncated by the end of the video | directly against the closing `</d>`, no space (HOUSE; the guides spell it `<cutoff>`, and **this divergence is unmeasured** — see below) |
| `<\|lyrics_start\|>` … `<\|lyrics_end\|>` | marks a sung run | **wraps** one or more `<d>` blocks (HOUSE) |
| `<\|caption_start\|>` … `<\|caption_end\|>` | undocumented; the name reads as subtitle | a **sibling** of `<d>`, immediately after the `</d>` it belongs to (HOUSE) |
| `<scenetrans>` | dialogue continuing across a cut | at the connecting point in **both** parts (GUIDE, base §4.4) — **but it matches no declared token** |

**The piped-versus-unpiped call is a decision, not a finding.** The release
declares `<|cutoff|>` and both guides print `<cutoff>`, so following the guide
literally gives the unpiped form and ours is the house call. **Nobody has
rendered the comparison.** It is reasoned from the release declaring one and not
the other, which is a good reason and not evidence; a downstream consumer
choosing the guide's spelling is following the vendor's text, not making a
mistake. Same standing for `<scenetrans>`, which matches no declared token at
all and stays ordinary prose.

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

**Whether a scene has a score is a decision about that scene, not a default.**
Reaching for `N/A` by reflex claims every clip you write is unscored, and
inventing a score claims something nobody asked for. Base §4.7 says only that
`N/A` is right when there is no non-diegetic music; the habit warning is ours,
and the examples in §10 write `N/A` to keep attention on the other fields.
[HOUSE]

**Mood words in this line are a recurring slip**, found across the bank on
2026-09-03 and recorded in [`prompt_audit.md`](../prompt_audit.md). The guide
sentence is stated and nothing checks it (§11).

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

