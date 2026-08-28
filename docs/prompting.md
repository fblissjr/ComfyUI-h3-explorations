# How to write an H3 prompt

last updated: 2026-08-28

The canonical house source. **Supersedes `internal/PROMPTING.md`**, which is now
working notes: it is gitignored, therefore reachable by no check, and it went
five days stale carrying a retracted claim and two instructions naming a node
that had been deleted. It still holds original material this file draws on — the
frame-grid work, the speech-budget derivation — so it is kept, not discarded.
Do not follow it directly.

Companions: [`prompt_catalogue.md`](prompt_catalogue.md) is what we currently
render (generated), [`prompt_audit.md`](prompt_audit.md) is whether those follow
these rules.

## Four layers, and every rule below says which one it is

Collapsing these is how two invented rules shipped here and were retracted.

| layer | what it is | breaking it means |
|---|---|---|
| **GUIDE** | the vendor's own text, `internal/official_prompt_guides/` base_en and ref_en | the prompt is **off-distribution** from what the model was trained on |
| **OWNER** | a deliberate design decision made here | not the vendor's, not a defect; cannot be cited as authority |
| **HOUSE** | our inference from measurement or experience | may itself be wrong; check before relying |
| **OPEN** | contradictory or unverifiable | do not build a checker on it |

**The guides are the only authority.** They are stable — base_en unchanged since
2026-08-21, ref_en since 2026-08-04 — which is why anything mechanically
checkable should derive from them rather than from any summary, this file
included.

**Guide text is not reproduced here.** `internal/` does not ship, so a reader
without it verifies by section number. The one exception is the closed
vocabulary below, restated because a rule nobody can mechanically check is
exactly how a broken prompt shipped.

---

## The mechanically checkable core

### Camera motion — GUIDE, base §4.3

A complete expression has three dimensions, and **all three draw from closed
sets**. Anything outside them is off-distribution.

Motion type: `Zoom In / Zoom Out`, `Push In / Pull Out`, `Pan Left / Pan Right`,
`Truck Left / Truck Right`, `Tilt Up / Tilt Down`,
`Pedestal Up / Pedestal Down`, `Arc Shot`, `Tracking Shot`, `Static Shot`,
`Shake Slightly / Shake Strongly`, `POV`,
`Roll Clockwise / Roll Counterclockwise`.

Amplitude: `with small amplitude`, `with large amplitude` — **and nothing else.**
Speed: `at slow speed`, `at fast speed` — **and nothing else.**

Two traps, both of which have shipped here:

- **Medium and normal are not expressions, they are omissions.** The guide says
  to add amplitude and speed only when meaningful, because medium amplitude and
  normal speed are the default. `at medium amplitude and moderate speed` is
  wrong twice — the words are not in the set, and the concept is "write nothing".
- **`Truck Left` and `Tracking Shot` are different rows.** `tracks left`
  conflates them.

Write motion as natural English action inside the shot, not as labels stacked at
the end — the guide gives worked examples at §4.3.

### Shot headers — GUIDE, base §4.2 and ref §5.1

`[Shot 1]` takes **no** timestamp. Later shots are `[Shot N] At MM:SS.mmm`.

A malformed header does not merely fail a check here — `bench/preflight_graph.py`
requires a literal `]` after the digits, so a header like
`[Shot 1, 00:00.000-00:06.000]` makes its shot list come back empty and **takes
three shot rules inert with it**. The grader is removed rather than reddened.

### Speaker identity — GUIDE, base §4.4

Identity goes **where the speaker first appears**, not where they first speak.
Introducing `(S2)` in one shot and describing them in the next is a violation,
and it is the second defect the shipped market prompt carries.

**Speaker ids are not conditional.** One of the v7 system prompts makes them
optional for a single unambiguous speaker; the guide and every guide example
carry them regardless.

### Sections and layout — GUIDE, base §2.2 and ref §1

The base modes take three fields; ref2va takes six sections in order. **The
label-on-its-own-line layout is ref-mode only** — applying it to the base modes
is a v7 defect, not a style choice.

### Reference labels — GUIDE, ref §5.2–5.4

`<Picture N>` / `<Video N>` / `<Audio N>` are **cited in
`detailed_description`**: §5.2's defining row, §5.3's frame anchors and §5.4's
audio citation all require it. Ref §2.2 separately says not to write a
standalone definition for an image supplying identity only.

---

## Owner decisions — not the vendor's, and not defects

**The four granular ref2va routes** — Entity, Scene, Attribute, Performance —
are an owner-authored selection layer. They are **not a fork**: all four and the
master emit ref §1's six sections in order with the label types, task types and
relationship markers intact. The routing axis has a guide anchor in ref §2.1's
four bullets for what `<Subject N>` covers.

Two consequences are worth knowing because they change output, not just routing:

- A `REFERENCE-RESOLUTION BOUNDARY` rule forbids `<Picture N>` and friends in
  `detailed_description`. That **inverts** ref §5.2, §5.3 and §5.4. It appears in
  the master too, so it is not a taxonomy artifact — it is a house rule that
  contradicts the guide, and the guide wins unless someone decides otherwise
  deliberately.
- Entity, Scene and Attribute pin the prefix to `[reference generation]`;
  Performance does not. Together with the rule above this makes
  `keyframe completion` **unreachable from three of the four routes.**

Two cut lines are judgement and not derivable from the guide: props sit under
Entity where ref §2.1 puts them in the clothing/props bullet, and Performance
collapses ref_en's deliberately orthogonal label-type and task-type axes into
one route.

---

## House rules — ours, and useful

From `internal/PROMPTING.md`, which the guides are silent on. Believe these
less strongly than anything marked GUIDE.

- The frame grid: duration snaps to `17k+5` at 24 fps.
- The speech budget, and turn-taking derived from it.
- The cast-sheet approach to keeping characters from collapsing into archetypes.
- Worth importing from v7: the closed-mouth endpoint rule for L2VA, the wardrobe
  grammar, and `"only"` as a strict exclusion.

---

## OPEN — do not build a checker on these

- **Turn cap.** `PROMPTING.md` §4.3 caps turns at one per three seconds; its §7
  block B says there is no cap. Both use a five-second clip as the example and
  give opposite answers, **and §7 is the text that gets pasted.**
- **No full stop before `<|cutoff|>`.** Stated unconditionally, but its
  mechanism — BPE dragging the `.` into the marker — described the pre-fix
  tokenizer path only. The rule is now unsourced rather than mechanical.
- **Closing punctuation on dialogue lines.** Block E states it universally; it
  is a ref §5.4 rule scoped to *reused source dialogue*, and as written it
  collides with the preserve-verbatim instruction directly above it.
- **Two numbers nothing here can check**: the 7,000-character cap, and the
  ref2va script anchor the whole speech budget rests on.
- **`<scenetrans>` and `<cutoff>` spellings** in the guides match nothing in the
  release's declared token list. A guide describing a mechanism whose token does
  not exist.

---

## Enforced by nothing

Stated so nobody mistakes silence for coverage.

- **Camera-motion vocabulary.** The denylist half is cheap and decidable and now
  has its escaped instance: the market prompt rendered badly and every gate
  passed it. See [`checks.md`](checks.md).
- **Speaker-identity placement.** Not mechanizable.
- **The mode-correct alignment sentence.** `preflight_graph.py` checks that a
  preamble names a Picture, not that it is the right sentence for the mode — so
  a mode-mismatched alignment line passes.
- **L2VA has no shipped prompt at all**, and
  `workflows/build_workflows.py`'s L2VA branch emits `[Shot N]` and `S.SS`
  unsubstituted where the fl2v path resolves both. The missing scene is hiding
  a live bug.
