> **Part of [`docs/prompting.md`](../prompting.md) (Governing Sources, Encoder Internals & Failure Modes (§14–§15)).**
> For the core prompt structure (`§1`–`§5.7`) and master router, see [`docs/prompting.md`](../prompting.md).

## 14. Every source that claims to govern a prompt, and where they disagree

Reconciled 2026-09-01. **Read this before citing any of them**, because they do
not carry equal weight and two of them are not authorities at all.

### 14.1 The sources, ranked by what a violation means

| # | source | on disk | standing | a violation means |
|---|---|---|---|---|
| 1 | the vendor's two guides | [`vendor_guides/`](../../vendor_guides/) | **the only authority** | the prompt is off-distribution from what the model was trained on |
| 2 | the vendor's own API payloads | `coderef/MiniMax-H3/scripts/readme/*.sh` (gitignored) | evidence of practice, not a rule | you are doing something the vendor's own pipeline never emits |
| 3 | the vendor's prompt-writing skill | `coderef/MiniMax-H3/.claude/skills/h3-prompt-writing/` | **a router, not a rule set** | nothing; it states no rule of its own |
| 4 | this file | `docs/prompting.md` | our reading, layered GUIDE / OWNER / HOUSE / OPEN | depends on the layer, which every rule names |
| ~~5~~ | ~~`internal/PROMPTING.md`~~ | **deleted 2026-09-01** | — | — |

**Row 5 is struck rather than removed.** `internal/PROMPTING.md` was ranked
here while it existed, and a reader who remembers being told to weigh it needs
to know it is gone rather than find it silently absent. **A ranked source that
does not exist is worse than an omitted one** — the row implies you could
consult it. Its content is in §14.5.

### 14.2 The vendor's skill adds no rule, and its guides are byte-identical to ours

MiniMax ship a prompt-writing skill twice, at
`.claude/skills/h3-prompt-writing/` and `.agents/skills/h3-prompt-writing/`.
Checked 2026-09-01:

- **The two copies are identical** (`diff -rq`, no differences).
- **Its `references/base-en.txt` and `references/ref-en.txt` are byte-identical
  to our `vendor_guides/` copies** — same size, same SHA-256.
  So our guide corpus is the vendor's own text, and that is now verified rather
  than assumed.
- **`SKILL.md` is 35 lines and defers entirely to those two files.** Its three
  "Output Rules" each trace to a guide statement — the English-body-with-original-
  language-dialogue rule to ref-en:5 and ref-en:217, the avoid-a-plot-summary
  rule to ref-en:7, the per-shot description checklist to ref-en:7.

**Why this check is trustworthy rather than lucky: BOTH files were hashed.**
Two artifacts agreeing is evidence only about the files you actually compared.
A bundle can carry one guide verbatim and a modified copy of the other, and
hashing the matching one and generalising to "this bundle is authentic" is a
conclusion the evidence does not support — the modified file is the one you
would then quote as vendor text. **Hash every file you intend to rely on, not a
representative of them.** That has been observed in the wild, so it is a real
failure mode and not a hypothetical.

**And if you re-derive a delta in Python, pass `autojunk=False`.**
`difflib.SequenceMatcher` treats "popular" elements as junk once a sequence
passes 200, and in a markdown file the popular element is the BLANK line — so a
stock `unified_diff` one-liner returns a valid but non-minimal edit script and
overstates the delta against what `diff` reports on the same pair. Verified
2026-09-01 on a 222-line against 252-line pair: 38 changed lines by `diff` and
by `autojunk=False`, 48 by the default. **`difflib.unified_diff()` exposes no
way to disable it** — use `SequenceMatcher(None, a, b, autojunk=False)` and walk
the opcodes. Reproduced only for line diffs of prose; an attempt to make it bite
on token-id sequences failed, so treat it as a prose-diff trap.

**So there is no fourth authority.** A session that finds the skill and reads it
as new guidance has found a second pointer to the guides we already have.

### 14.3 Where our shipped prompts diverged, and what happened to each

**Shot line breaks — CLOSED 2026-09-01, we conformed.** Every vendor
multi-shot specimen runs shots inline in one unbroken paragraph, and a peer
session found the same in a third, third-party corpus: it is inline in every
multi-shot specimen that follows the official three-field grammar, and breaks
lines only inside a four-section format of that corpus's own invention.
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
| shot continues | ours 32, 3rd-party 17, vendor 2 | ours 12, 3rd-party 7, vendor 4 |
| line ends the shot | ours 0, 3rd-party 0, vendor 0 | ours 1, 3rd-party 16, vendor 1 |

**The provenance matters and is weaker than it looks.** The pattern is a
third-party corpus observation that our corpus happens to satisfy — our mid-shot rate is
73% against their 71%. **The vendor corpus does not corroborate it**: its
decisive shot-final cell holds a single line, which is consistent with the rule
and establishes nothing, and its mid-shot cell runs the *other* way at 2 of 6.
So do not cite the positional rule as vendor practice. The only STATED vendor
rule here remains base-en:136, for voiceover, which nothing contradicts, and
§5.6's [HOUSE] label was right all along.

It is still the right STATISTIC — it is what turned an apparent 43% cue rate
into zero counterexamples and retracted the misalignment — but a statistic being
right does not make its source authoritative.

**Turns per shot — CLOSED 2026-09-01 by the owner, on the render.**
`DIALOGUE_T2V_PROMPT` puts four `<d>` blocks in `[Shot 1]` and three in
`[Shot 2]`, where every base-format vendor specimen uses one. **The owner's
verdict on the rendered scene is that it clearly works**, so stacked turns are
an OWNER-sanctioned capability here, not a divergence to correct.

**Why one render settles this and does not settle most things.** "Does the
model deliver four turns in a shot at all" is a presence/absence question, and
those a single clip can answer because the behaviour either occurs or does not.
It is NOT evidence that stacking is *better* than cutting — that would be a
comparison, and a comparison needs matched seeds and a distribution
(`docs/eval_comparison.md`). Nobody has run that, and nothing here claims it.

A first pass also got the vendor evidence wrong and it is worth keeping: it
recorded "the vendor uses exactly one `<d>` per shot in every specimen", which
was **wrong, by counting shot headers across a whole prompt rather than inside
the narrative field**. The vendor's reproducible ref2va payload puts TWO turns
in ONE shot; its second `[Shot 1]` string is a cross-reference inside
`retention_analysis`, not a header. So the reference format was never unanimous,
and the ref2v scene arms at two turns were doing exactly what the vendor does.

`bench/diff_prompt_corpus.py` still reports the base-format difference. **Read
that row as a description, not a defect** — it is the tool doing its job, and
the answer to it is this paragraph.

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

### 14.5 Where `internal/PROMPTING.md`'s content went

**That file is DELETED as of 2026-09-01.** This section used to be headed "what
it still uniquely holds" and named §7 and §7b -- its LLM prompt-writer system
prompts -- as content that had NOT moved and as "the only reason to open the
file". **Both halves are now wrong, and left here rather than reworded because
the section spent a day telling readers to open something that no longer
exists.**

What moved into this file: §1's model facts and the Context-IR framing, §2's
presentation contract, §3's five diagnoses, §4's frame grid and speech budget
including the open two-parameter problem, §5's documented-to-work list, and §6's
user-message contract. All of it is §15.

**What closed the §7/§7b gap:** [`docs/portable/h3_system_prompt.md`](../portable/h3_system_prompt.md),
a writer-model system prompt — a CORE section plus one block per mode — derived
from the portable standard and tracked. It is a draft and says so. Its Part One
strings are graded against the guide-parsed constants by
`bench/check_prompt_docs_sync.py`, so it cannot drift silently, but **"grades
clean" there covers the STATED mechanical rules only** — a control breaking a
GUIDE-SHOWN rule correctly stays green, because the grader refuses to enforce
shown rules. Most of that file is untested by construction.

**A sibling set of per-mode drafts existed at
`internal/prompts/2026-08-22_*_system_prompt.md` and was REMOVED by the owner on
2026-09-01.** They were externally authored, corrected against the guides, and
covered four modes — t2va, i2va, fl2va, ref2va, with **no l2va draft**, so the
set never had parity.

**Recorded so the next reader does not go looking.** `internal/` is gitignored,
so nothing there survives in history and nothing there should be cited as though
it will still be present. **If you are about to depend on a file under
`internal/`, copy what you need into the tracked tree first** — that is the
lesson the vendor guides taught this morning, and the same day proved it twice.

## 15. What the model is, and what actually reaches it

**Removed and restored the same day.** `a52999d` (2026-09-01), whose message
describes a rewrite of §14.5 and the striking of two ranking rows, also dropped
this whole section, while the §14.5 it wrote says the migrated content "is
§15". Restored verbatim from that commit's parent later the same day, and it
has stood since; `docs/portable/h3_system_prompt.md` routes readers here.

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
- **Core passes your prompt through verbatim.** `tokenize_with_weights` ends with
  `add_text(text)` (`comfy/text_encoders/minimax.py:197`) after every label; nothing is stripped,
  parsed, reformatted, or reordered.
- **Our conditioning nodes strip the two ends first**, through
  `h3_rules.normalize_prompt` (`MiniMaxH3Conditioning`,
  `MiniMaxH3ReferenceConditioning`; checked 2026-10-01 against that file). A
  trailing newline is one more token and one more token is a different sample at
  the same seed, so a prompt pasted into the UI with one now tokenizes like the
  graph that ships it. Interior newlines are kept, because they carry the
  prompt's structure. Core's own H3 nodes do not strip.
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

### 15.3 The seven ways prompts go wrong here

Carried forward from `internal/PROMPTING.md` §3. These are diagnoses from
observed outputs, not measurements. Items 6 and 7 were added on 2026-09-20 from
the bank read of 0.128.0; **neither vendor guide states them**, which is why
both carry a HOUSE tag and an arm that rendered the defect.

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
   paces beats by feel and overruns the clip. §3.3 is the fix. (Until 2026-09-18 this read "picks cut timestamps"; the house no longer writes them, section 3.1.)
4. **Line breaking has no single convention** and the vendor's own artifacts use
   two — see §12.11 and §14.3, which is where this now lives.
5. **Characters talk over each other**, from three causes: two speakers in one
   shot with nothing to order them but prose (a cut is the only hard separator
   the format offers); a **compound speaker id used by mistake** — `(S4,S5)` is
   the documented notation for literal simultaneous group speech, so it
   instructs two characters to say the line in unison; and simply too many
   speakers for the runtime. Every on-screen character not given an explicit
   "produces no vocal sound" is a candidate for the model to voice anyway.
   **Reserve that phrase for a character who never vocalises in the clip**, the
   category base §4.4 names when it says characters who never vocalize receive
   no id. Someone silent in this moment who speaks later is written as what they
   do instead — "does not answer". [HOUSE; the phrase appears in neither guide]
6. **An action with no agent renders with no cause, or the model invents one.**
   The covered market prompt had coins clattering into a tin with nobody
   dropping them, and arms rendered them arriving from nowhere and out of a
   crate; a breaker slammed itself, an envelope pushed itself out. Name who does
   it. The guides require every detail to correspond to something visible or
   audible (base §4.1) and an uncaused sound does correspond to something
   audible, so this is a reading of renders, not of the guide. [HOUSE, observed;
   `bench/results/2026-09-18_sol_reorder_panel.md` and CHANGELOG 0.127.0]
7. **A count is a promise the rest of the scene has to keep.** "A plate of
   hamburgers" in front of each child rendered two burgers each; a second crate
   nobody had lifted appeared anyway; plates "in a row" on a two-cover ticket.
   Write the number you want to see, then check it against what the shot
   actually shows. Neither guide states anything about quantities. [HOUSE,
   observed; CHANGELOG 0.126.1 and 0.128.0]

Items 6 and 7 are two members of one class, a slot the prompt left open that the
model filled. §16 lists the rest (the cast count, who is in each shot, one
geography) and says how to choose which slots to pin.

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

**One system prompt can serve several modes** if it names the active mode,
says "apply only the active mode", and keeps each mode's alignment line and
layout beside it; the writer then emits only that mode's alignment format.
That is how the external writer prompts handle I2VA and L2VA in one file.
[HOUSE]

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
- **Animation on twos.** At 24 fps, animation on twos targets about 12 unique
  states per second; the external writer prompts state the exposure cadence as
  a target rather than leaving it to the style word.

---

