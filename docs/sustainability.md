# Keeping this repo worth working in

last updated: 2026-08-28

What to do, and deliberately not do, to keep this repo delivering. Written after
a survey of the custom nodes against five other H3 implementations
([`custom_node_gaps.md`](custom_node_gaps.md)) turned up more about *this
repo's* shape than about the comparison, and revised after an adversarial review
broke four of its claims. What survived is below; what did not is recorded at
the end, because the failures are more instructive than the argument.

**This file owns no facts.** [`roadmap.md`](roadmap.md) owns what to work on
next and what would count as finding it; [`checks.md`](checks.md) owns the
standard and the uncontrolled-requirements audit;
[`open_experiments.md`](open_experiments.md) owns what is deliberately not
measured. This is a direction argument that defers to all three.

**Retire it when it stops being argued with.** A direction document nobody
disagrees with has either been absorbed or ignored, and both mean delete.

---

## The finding

**The instrumentation is healthy. The closure rate is not.**

Every mechanism here prevents *false belief*: the evidence table, the
"enforced by nothing" column, retraction tracking, the do-not-rely-on lists, the
different-sample rule. That is rare, and it is why numbers from this repo can be
trusted.

What is missing is any **general** forcing function that makes an open question
close. Two local ones exist and both work, which is the reason to generalise
rather than invent:

- [`research/pdd/queued_arms.md`](research/pdd/queued_arms.md) already says
  *"Delete an entry when it runs; one still here in a week belongs in one of
  those two instead."* That is an aging rule, in-tree, and it is the model.
- [`roadmap.md`](roadmap.md) closes items with dates, and `checks.md` has a
  retirement convention for checks.

Neither reaches the big trackers. The evidence that this matters is
`open_experiments.md`'s own history: sampled across commits, its numbered
sections went **7 → 8 → 13 → 16 → 19 → 21 → 24 → 24** — monotone at every
sample, with 21 of 23 still open. That is the shape of a list nothing ages.

The risk is not that this repo will be wrong. It is that it will be
indefinitely, accurately undecided.

## Do this

Ordered. The first two are worth more than the rest together.

### 1. Generalise the `queued_arms.md` rule to the big trackers

Give every open question a review date, not just a status. When the date
passes, "we are living with this, reviewed `<date>`" is a legitimate and
*different* state from open.

This adds no instrument. What it changes is that leaving something open becomes
an act rather than a default. `queued_arms.md` shows the convention already
works here; the gap is that it covers one lane.

### 2. Close the top of the backlog before adding capability

Ranked by cost-of-being-wrong times cheapness-to-settle.

| arm | why now | what it needs |
|---|---|---|
| encoder bounds — weights fixed, bounds varied | shipping on one render's evidence; three other implementations disagree with us | a reachable path; the module entry point exists |
| a PDD blind session | shipping and never judged perceptually | a seeded session per [`eval_comparison.md`](eval_comparison.md) |
| W4 DiT arch probe | closes or opens a lane; the kernels are in the installed wheel | minutes |
| SLA router coverage of the token refiner | the LoRA adapts modules the router does not patch | a read, then a decision |
| video VAE default | **already graded** — `bench/results/2026-08-21_vae_encoder_precision.json`, with probe arms wired | a decision, not a measurement |

`roadmap.md` owns this queue. If it disagrees, it wins.

**The last row is a correction, and it is the useful one.** An earlier draft
called the VAE precision node unwired and asked for a render to settle it. It is
wired (`h3_probe_ref_vae_encoder_fp32*` against a matched fp16 control), the
answer was graded on 2026-08-21, and
`bench/grade_vae_encoder_precision.py` opens by explaining that a rendered pair
*cannot* A/B a numerical knob — so the draft asked for the one experiment its
own repo forbids. Three errors stacked, from one dropped scope.

### 3. Make the rest of `bench/` visible

`check_doc_inventory.py` differences two directory listings for `check_*.py`.
Everything else there — 106 non-check `.py` files as of 2026-08-28, across five
subdirectories, of which only `bench/red/` is cited in `checks.md` — is indexed
by nothing.

**A report, not enforcement, and this is a retreat from the previous draft.**
That draft called this "the one place worth adding enforcement" on a
hypothetical, ten lines after refusing other checks for lacking an escaped
instance. There is no escaped instance here, so by this repo's own rule the
answer is a listing someone reads, and a widening of `check_doc_inventory.py`
past the scope its docstring drew deliberately is not justified yet.

### 4. Give the node registry a retirement state

`node_id` is append-only by rule and must stay — a saved graph binds to it. But
**registered, wired nowhere, kept for externally saved graphs** is a real status
and should be declared rather than rediscovered by a survey. The count depends
on whether bench graphs count, which is itself the argument for declaring it.

### 5. Derive indexes; date claims

[`wiki/index.md`](wiki/index.md) is the template — it cannot drift because it is
regenerated from tables its owners maintain. Where prose must assert a value,
prefer citing the command that prints it.

The drift class this attacks is the one that recurs. On 2026-08-28 alone: a
generator comment stating the opposite of the graph it emits, a comparison
document overtaken by upstream, four stale statements in a lane document, a
false claim in a sibling pack's README — and this file's own VAE error. Every
one was prose sitting next to the mechanism it described, disagreeing with it.

## Do not do this

Each is a plausible next step the repo's own rules refuse. The adversarial
review could not break any of them.

- **A documentation freshness gate.** It would report red while the state is
  correct, which trains people to ignore red.
- **Checks for the "enforced by nothing" rows without an escaped instance.**
  The label *is* the coverage.
- **A coverage target for node registration**, or any ratio that looks better
  without changing what a reader does next.
- **A second index of the documents.** There is one, it is generated, and a
  hand-maintained companion is the exact failure this file argues against.

**And do not add a "behavioural probe" check class.** The previous draft
permitted exactly one new class on these grounds and was wrong twice over.
That class is already installed at least three times —
`check_conditioning_behaviour.py`, `check_reference_contracts.py` and
`check_reference_runtime.py` all drive real nodes against real upstream imports
on CPU with stubs. And the escaped instance offered as justification would not
have been caught by what was prescribed: `bench/count_packed_rows.py` already
constructs the upstream object with stub shapes and would have stayed green,
because the break was in *our call site* passing an argument upstream had
removed, not in the constructor. **The gap is one node, not one class** — and it
is a call-site signature question, which is a different and much smaller thing.

That is the 2026-08-17 failure mode verbatim: a proposed instrument dissolving
on contact with one already installed. It is recorded here rather than deleted
because this file argued for care about exactly that, and then did it.

## What the review broke

Kept as a record, because a direction document that hides its corrections is
worth less than one that shows them.

- The VAE recommendation, above — wrong in three ways from one dropped scope.
- The permitted new check class — already installed, and misjustified.
- **A `1.8x` checks-to-code ratio**, deleted rather than fixed. Its denominator
  excluded `workflows/` (a generator of comparable size); including it gives
  roughly parity. The defence offered for it — that check files are mostly
  docstring rationale — was tested with an AST pass and **refuted**: they are
  *more* executable than the node code, which moves the ratio the other way. And
  it sat beside "do not economise here", which no plausible substitute value
  would change, so it failed this repo's own substitution test and was
  decorative from the start.
- Several surface counts, corrected or removed. The pattern in the wrong ones:
  a recursive file count paired with a non-recursive line count, and a
  four-document total that double-counted rows meaning different things.

The thesis survived. Nearly every number supporting it did not, which is a fair
description of how this file came to exist.
