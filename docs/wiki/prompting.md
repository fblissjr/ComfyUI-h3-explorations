# Prompting H3: where all of it comes from

last updated: 2026-09-01

**A router, not an authority.** It states no prompting rule of its own. Where it
disagrees with an owner below, the owner is right.

This page exists to answer one question without a search: *when we say something
about how to prompt H3, where did it come from, and how much does it bind?*

---

## 1. The five sources, ranked

Only the first binds. Two of the five are not authorities at all, and both have
been mistaken for one.

| # | source | standing | a violation means |
|---|---|---|---|
| 1 | **the vendor's two guides** — `internal/official_prompt_guides/base_en`, `ref_en` | the only authority | the prompt is **off-distribution** from what the model was trained on |
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

## 6. Open, and recorded rather than fixed

Neither breaks a **stated** guide rule. Both are unenforced and easy to mistake
for settled; `prompting.md` §14.3 owns them.

- **Shot line breaks.** Every vendor multi-shot specimen runs shots inline in one
  paragraph. Of our multi-shot t2va prompts, only `DIALOGUE_T2V_PROMPT` does.
- **Dialogue turns per shot.** Our shipped dialogue default puts four `<d>`
  blocks in one shot; `internal/PROMPTING.md` §4.3 asks for one. The guides state
  no limit. Unrendered, so this is two of our own documents disagreeing rather
  than evidence about the model.

**The distinction doing the work in both:** a guide **statement** is a rule; a
guide **example** is not. Two rules have been invented here by reading examples
as rules, and both were retracted. `prompting.md` marks every GUIDE rule *stated*
or *shown* for exactly this reason.
