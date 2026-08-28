# Why `h3_config.py` drifts, and what would actually stop it

An audit on 2026-08-28 found roughly two dozen stale or contradicted claims in
`workflows/h3_config.py`. Six were flatly wrong about what currently ships and
have been corrected; the rest are recorded below by class rather than
individually, because **the classes are the useful output and the instances are
not.** This file argues about the mechanism. It asserts nothing about H3.

## What was actually found

Not "the comments got old". Four distinct failure modes, and they need
different fixes:

| class | example found | why it happened |
|---|---|---|
| **Ship state written into prose** | `ENCODER_V2` opened "the shipped text encoder … v2 since 2026-08-27"; `ENCODER_V1` opened "shipped again since 2026-08-27". Neither ships; `MODELS["clip"]` is `ENCODER_INT8` | the fact has a machine-readable home three lines away, and was ALSO written in English |
| **Decisions recorded as rules, then reversed** | "`REF_VIDEO_LENGTH` was deleted. Do not reintroduce it" — reintroduced six days later, defined ~100 lines below, 28 graphs render at it | the reversal updated the constant and not the prohibition |
| **Claims of absence that a later run filled** | "no sage kernel has been graded against either … no number to quote" — the grading exists, in a dated record, produced by a named script | absence claims are true when written and silently expire |
| **Arithmetic restated in prose** | "1024x1536 costs 1.78x the attention" (that is the TOKEN ratio; attention squares it, ~3.2x) and "every ratio ≥1.75 costs the SAME" (7% spread; the four canvases named were selected) | a number retyped beside the rule that generates it |

One instance deserves singling out because it is the mechanism in miniature:
inside `ENCODER_INT8`'s own comment block, one line said the shipped graphs load
v2 and another said this file ships on every graph. **`git blame` shows a single
commit wrote both.** Nobody introduced a contradiction over time; it arrived
contradictory. That rules out "review more carefully" as the fix.

## The actual mechanism

`h3_config.py` is ~1,600 lines, of which the overwhelming majority is prose.
That is not a flaw — the prose is why this repo can be picked up mid-thread, and
several comments in it are the only surviving record of why a value is what it
is. The problem is narrower:

> **A constant and its explanation are one edit. A constant and a *different*
> constant's explanation are two, and only one of them is enforced.**

Every failure above is a change that updated a value while leaving prose
elsewhere describing the old value. Nothing is wrong with the prose as a
practice; what is wrong is that **prose about a fact that has a machine-readable
home is a second copy of that fact**, and this file's own opening rule already
forbids second copies — for values. It does not extend the rule to sentences,
which is exactly where the drift lives.

## What will not fix it

Worth stating, because these are the reflexes and this repo has already spent a
day discovering that instruments dissolve on contact with what is installed
(`docs/sustainability.md`).

- **Review discipline.** Refuted by the single-commit contradiction above.
- **Deleting the prose.** It is load-bearing. Several comments are the only
  record of why a tolerance sits between two noise floors, and one of them
  prevented a correct bake being silently rejected.
- **A general "docs match code" checker.** Undecidable in the general case, and
  the failure mode is worse than the disease: a checker that reports red while
  the state is correct trains you to ignore red, which `docs/checks.md` already
  records as the reason two existing checks use allowlists.
- **More dates.** Every wrong claim above already carried a date. The date told
  you when it was written, not whether it is still true.

## What would

Ranked by leverage per unit of work.

### 1. Derive ship state instead of describing it

The single highest-value change, and it generalises past this file. Sentences
like "shipped on every graph since X" are **queries**, not facts: the answer is
in the graphs, and the graphs are generated. A constant's docstring should say
what the value MEANS and why it was chosen — never what currently uses it.

Concretely: a `bench/` reporter that prints, for each entry in `MODELS`,
`CANVAS_TIERS`, and the LoRA sets, how many shipped graphs reference it and
which. Then the docstrings say "see the reporter" and stop asserting counts.
This is the same move `docs/prompt_catalogue.md` already makes — generated from
the graphs, judges nothing — and it is why that file has not drifted.

**This is the general answer to "make the code the source of truth".** Not
"delete the prose": *stop writing down anything a script can answer, and write a
script for the things worth answering.* The prose that survives is the part that
explains a decision, which no script can reconstruct.

### 2. Give absence claims an expiry that is a test, not a date

"There is no number to quote" is a claim about the filesystem. It can be
checked: if `bench/results/` gains a record matching the thing the comment says
does not exist, the comment is stale. That is a narrow, decidable check —
unlike "docs match code" — because the claim names an artifact.

Cheapest form: a convention that absence claims cite the glob that would refute
them (`no record matches bench/results/*sage*capture*`), plus one check that
walks those globs. A claim that cannot name what would refute it should not be
written as a fact.

### 3. Split the file along its actual seam

Two populations live here with different lifetimes: **values the generator
reads** (short, stable, load-bearing) and **the record of why** (long, historical,
append-only). They are interleaved, so a reader scanning for the first is served
paragraphs of the second, and an editor changing the first does not see the
second scroll past.

Not proposed as a mechanical split — the adjacency is genuinely useful when the
prose is one paragraph. Proposed for the blocks where the prose exceeds, say, a
screen: move the history to `docs/`, leave a one-line pointer. `SOL_*` and the
encoder constants are the obvious candidates; between them they are several
hundred lines of narrative around a handful of numbers.

### 4. Enforce the file's own no-second-copy rule

`h3_config.py` opens by saying nothing in it may have a second copy anywhere in
the repo. **That is enforced by nothing, and is currently violated** — the seed
literal `730451892` is retyped in six `bench/` scripts, two as their own
constants, and `LONG_LENGTH`'s 362 is retyped alongside it.

By this repo's own standard the rule should either name the assertion that goes
red or say "enforced by nothing". A grep-based check for the handful of
distinctive literals (the seed, the canvas strings, the model filenames) is
decidable and small. Note the precedent it follows: `REF_QWEN_SHORT_EDGE` was
moved to `h3_rules.py` on 2026-08-28 precisely so the node default and the
generator could not disagree — that is this rule applied by hand, once.

## The one-line version

**Prose that states a fact the code already knows is a cache, and this repo has
no invalidation for it.** Either derive the fact, or write the sentence so that
it explains a decision rather than reporting a state — decisions do not go
stale, states do.
