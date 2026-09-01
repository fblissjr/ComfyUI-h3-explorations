# Prompting H3: where all of it comes from

last updated: 2026-09-01

**A router, not an authority.** It states no prompting rule of its own. Where it
disagrees with an owner below, the owner is right.

This page exists to answer one question without a search: *when we say something
about how to prompt H3, where did it come from, and how much does it bind?*

---

## 0. If you are an agent, start with the skill

`.claude/skills/h3-prompt` routes by TASK — writing one, editing a shipped one,
converting between modes, judging whether one is good — to the file that owns
each answer and the command that verifies the result. It restates no rule, so it
cannot drift into a second authority.

**This page and that skill exist for different readers.** The skill is what an
agent hits when it says "I need to write an H3 prompt". This page is the map: it
answers where a claim came from and what guards it.

### The whole lane, and what guards each edge

| from | to | how, and what guards it |
|---|---|---|
| vendor guides | alignment templates, camera vocabulary | **parsed at import** by `preflight_graph.py` / `check_prompt_guide_conformance.py` / `check_camera_vocabulary.py`; a missing guide is a loud error, never a silent pass |
| `build_workflows.py` | `workflows/*.json` | generated; graphs are never hand-edited, and nothing is true of a graph until rebuilt |
| the graphs | `prompt_catalogue.md` | generated; `build_prompt_catalogue.py --check` |
| catalogue scenes | `prompt_audit.md` verdicts | hand-written, **coverage checked** by `check_prompt_docs_sync.py` |
| vendor guides | `prompting.md`'s quoted vocabularies | retyped for readers, **checked per instance** |
| `prompting.md` §10 | still grades clean | **re-run through `preflight_graph.grade`** |
| `prompting.md` §10 | the portable standard's examples | **checked verbatim** |
| `CLAUDE.md` | `docs/wiki/index.md` | generated; `build_wiki_index.py --check` |
| dated snapshots | frozen | pinned by hash; **never** graded against current sources |
| shipped prompts | vendor practice | `diff_prompt_corpus.py`, a report rather than a gate |

**Enforced by nothing, and deliberately:** the manual's prose — the rules
themselves, their layer assignments, the reasoning. No mechanism can diff those.
What catches an error there is a second implementation computing it fresh, not a
second reading.

## 1. The five sources, ranked

Only the first binds. Two of the five are not authorities at all, and both have
been mistaken for one.

| # | source | standing | a violation means |
|---|---|---|---|
| 1 | **the vendor's two guides** — `vendor_guides/base_en.md`, `ref_en.md` | the only authority | the prompt is **off-distribution** from what the model was trained on |
| 2 | **the vendor's own API payloads** — `coderef/MiniMax-H3/scripts/readme/*.sh` | evidence of practice | you are doing something the vendor's pipeline never emits |
| 3 | **the vendor's prompt-writing skill** — `coderef/MiniMax-H3/.claude/skills/h3-prompt-writing/` | **a router; states no rule** | nothing |
| 4 | [`docs/prompting.md`](../prompting.md) | our reading, every rule layered | depends on the layer, which the rule names |
| 5 | `internal/PROMPTING.md` | **superseded, being retired** | nothing — cite source 4 |

Sources 1, 2, 3 and 5 are gitignored and do not ship, which is why they are named
in backticks rather than linked.

**Two facts worth not re-deriving**, both established 2026-09-01:

- **The vendor's skill adds nothing.** Its bundled `base-en.txt` and `ref-en.txt`
  are byte-identical (SHA-256) to our copies, its two locations are identical to
  each other, and its `SKILL.md` is 35 lines that defer to those files. Its three
  "Output Rules" each trace to a guide sentence. **Finding it is not finding a
  new authority.**
- **Our guide corpus is therefore the vendor's own text, verified rather than
  assumed.**

---

## 2. Who owns which question

| question | owner |
|---|---|
| how do I write a prompt, in any mode | [`docs/prompting.md`](../prompting.md) — **start here, it is the single source of truth** |
| what does this repo actually render | [`docs/prompt_catalogue.md`](../prompt_catalogue.md) — generated from the graphs; judges nothing |
| is what we render any good | [`docs/prompt_audit.md`](../prompt_audit.md) — hand-written, one verdict per scene |
| what do the reference labels mean, and how are images sized | [`docs/h3_references.md`](../h3_references.md) |
| what canvases and lengths are legal | [`docs/h3_resolutions.md`](../h3_resolutions.md), [`docs/h3_geometry_and_nodes.md`](../h3_geometry_and_nodes.md) |
| how do I compare two prompts fairly | [`docs/eval_comparison.md`](../eval_comparison.md) — and read the warning in §5 below first |

**The catalogue is generated and nothing regenerates it**, and it has been stale
before. Run this before trusting it:

    python bench/build_prompt_catalogue.py --check

---

## 3. At least five worked examples per mode

`docs/prompting.md` §10 carries worked examples for every mode, and **every one
is graded** — see §4. Shipped prompts are additional examples, but they are *what
we render*, not *what is exemplary*: read them through the audit's verdict.

| mode | worked examples in §10 | shipped |
|---|---|---|
| **T2VA** | §10.1 | see [`prompt_catalogue.md`](../prompt_catalogue.md) |
| **I2VA** | §10.2.1 – §10.2.5 | as above |
| **FL2VA** | §10.3.1 – §10.3.5 | as above |
| **L2VA** | §10.4.1 – §10.4.5 | as above |
| **Ref2VA** | §10.5 | as above |

The shipped column is deliberately a pointer. How many scenes exist per mode is
a property of the graphs, and a number written here would be a second copy of it
with nothing to invalidate it. `bench/grade_prompt_text.py --list-donors` names
the canonical graph per mode; the catalogue lists them all.

**The keyframe modes are where the examples were needed.** Each shipped exactly
one prompt, so a reader had one specimen and no sense of the range. The four new
ones per mode deliberately vary shot count, dialogue presence, register and
camera type — and §10.3.3, §10.3.5, §10.4.3 and §10.4.5 are the only multi-shot
keyframe prompts that exist anywhere in this repo.

---

## 3b. Handing the standard to someone outside this repo

**Published at [https://claude.ai/code/artifact/daa73be5-0bb7-4d02-8456-5dc107acea54](https://claude.ai/code/artifact/daa73be5-0bb7-4d02-8456-5dc107acea54)**, source at
[`docs/portable/h3_prompt_standard.html`](../portable/h3_prompt_standard.html).
Republishing that same file keeps that URL, so the link is stable and can be
handed out once.

It is a **self-contained** extract: the rules with their layer, the per-mode
structure, the operational answers on speaker IDs and markers, two graded
examples of what good looks like, and nine failure patterns with why each one
is wrong. **It cites no repository path**, so it survives being read by an agent
in another repo or pasted into a ticket.

It is derived from `prompting.md` and says so on the page; where the two
disagree, `prompting.md` is newer.

**It is a second copy of rules owned elsewhere, so it is checked rather than
trusted.** `bench/check_prompt_docs_sync.py` verifies every QUOTATION on it
against the source that owns it — the three Part One templates against the
guide-parsed constants, the camera vocabulary against the guide's own table,
every worked example against §10 verbatim, and each quoted guide sentence
against the guide. **Run it after editing either file.** Its prose is not
checkable and is not checked; a green run means the quotations still match.

**Dated snapshots live beside it**, e.g.
`docs/portable/2026-09-01_h3_prompt_standard.html` — a frozen record of what was
published and shared on that date, carrying a banner saying so. **A snapshot is
never graded against current sources**: it will fall behind the manual by
design, and grading it would go red for the one reason that is correct. It is
checked only for having stayed unmodified, against a hash in
`docs/portable/snapshots.json`. A snapshot-shaped file that nothing records
also fails, because it would look authoritative with nothing pinning it.

It had drifted before it was checked once: the T2VA example was transcribed
with `non_diegetic_music: N/A` where the manual carries a real cue — into the
exact habit the same page warns against — and seven camera motion types had
lost half their name to abbreviation.

## 4. Grade a draft before you render it

    python bench/grade_prompt_text.py --mode fl2va --length 345 draft.txt

Wraps loose prompt text in a shipped graph of that mode and runs
`preflight_graph.grade`. It adds no rules; it makes the existing grader reachable
for text that is not in a graph yet. Nonzero exit on FAIL.

Two things that will otherwise waste an hour:

- **A prompt is conformant AT A DURATION.** `S.SS` in the Part One line and every
  `At MM:SS.mmm` cut resolve against the snapped frame count, so the same text is
  correct at 345 frames and wrong at 192. Always pass `--length`.
- **For Ref2VA, pass `--like` a graph wiring the references your prompt
  declares.** Labels are graded against the donor's sockets, so a prompt naming
  two pictures and an audio clip fails against a one-picture donor — a property of
  the donor, not the prompt. The tool says so when every failure is that shape.

For a prompt already in a graph, use `bench/preflight_graph.py <graph.json>`
directly. It reports and never refuses.

---

## 5. What is known to go wrong

Each is owned elsewhere; this is the index.

| | where |
|---|---|
| the model renders what is **described**, not what is **named** — expand every name into a visual noun phrase once and reuse it verbatim | `prompting.md` §15.3 |
| **nothing injects the duration** into your prompt, so a writer picks cut times by feel and overruns the clip | `prompting.md` §15.2, §3.3 |
| the speech budget's **shape is unsettled** and may be 1.8x too tight on short shots — do not cite it as measured | `prompting.md` §15.4 |
| on-screen text must be **typed literally**, or the model draws the texture of English | `prompting.md` §6 |
| `(S4,S5)` means **literal unison**, not "and also this character" | `prompting.md` §15.3 |
| there is **no negative prompt channel** — guidance is CFG-distilled | `prompting.md` §15.1 |
| **camera-motion vocabulary is enforced by nothing** | `prompting.md` §13 |

**The trap that outranks all of these: a rendered clip cannot A/B a prompt
change.** Two arms differing in any way draw different samples, not better and
worse versions of one — they diverge at frame 0. "Which clip looks better" is a
draw from a distribution. A perceptual claim needs many seeds per arm judged
blind in aggregate ([`docs/eval_comparison.md`](../eval_comparison.md)), and one
clip per arm cannot support it however carefully it is stacked.

---

## 6. Open, closed, and withdrawn

`prompting.md` §14.3 owns all three; `bench/diff_prompt_corpus.py` re-derives them.

- **Shot line breaks — CLOSED.** Vendor practice and a third-party corpus both
  run multi-shot prompts inline, with no counterexample. Ours were split; they
  were conformed on 2026-09-01. Clips of the changed scenes from before that
  date are not matched-seed comparable with clips after it.
- **Mouth closing — WITHDRAWN.** The rule is positional (cue when the shot
  continues, never when the line ends the shot), and we already followed it. The
  earlier finding was a per-line ratio, which is the wrong statistic. The
  positional pattern is a third-party observation, **not vendor practice** — the
  vendor corpus does not corroborate it.
- **Turns per shot — OPEN, base format only.** Vendor base specimens use one
  dialogue block per shot and `DIALOGUE_T2V_PROMPT` does not. The vendor is
  SPLIT in reference format — its own payload puts two turns in one shot — so
  our ref2v scene arms are within vendor practice, not outliers. No guide states
  a cap and nothing has been rendered.

- **Two house patterns — NOTICED, not defects.** We write `N/A` for
  `non_diegetic_music` and a one-sentence `overall_soundscape` far more often
  than the vendor does. Both legal; both decisions nobody actually made.

**How these get found at all:** `bench/diff_prompt_corpus.py` runs two passes.
The first reports every feature where the vendor never varies and we do. The
second reports features where BOTH corpora vary but our rate sits far from
theirs — a house pattern rather than a divergence. The second pass exists
because the first cannot see a skew however extreme: it suppresses any feature
the vendor varies at all, which is correct, and which is why our N/A rate went
uncounted through repeated runs. That is the shape of all three. Neither
`preflight_graph.py` nor `check_prompt_guide_conformance.py` could have found
them — one encodes the stated rules, the other refuses to assert anything the
guide does not state, and these are all things the vendor does consistently and
never says.

**Its limit is the real one:** the feature list is hand-authored, so it finds
only divergences someone thought to encode. Three of its own statistics were
wrong before they were right, and each crude version hid the finding its sharp
version surfaced. Read a clean report as "nothing on the axes we thought of".

**The distinction doing the work throughout:** a guide **statement** is a rule; a
guide **example** is not. Two rules have been invented here by reading examples
as rules, and both were retracted. `prompting.md` marks every GUIDE rule *stated*
or *shown* for exactly this reason.
