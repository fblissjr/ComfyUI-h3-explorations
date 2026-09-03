# What this repo holds itself to

This is a research hub for running and studying MiniMax H3 inside ComfyUI on
one box. Its output is not the nodes or the workflows. Its output is claims
about the model that a later reader can re-derive, and the tooling that makes
re-deriving cheaper than trusting.

This file is the why. It should rarely change. `CLAUDE.md` is the operative
rules and the instances that earned them, and it changes constantly. `docs/`
carries the stories. If a tenet here turns out to be wrong, change it in a
commit that says what it used to claim and what refuted it.

## Truth

**Code is right when prose disagrees.** A sentence stating a fact the code
already knows is a cache with no invalidation. Cite the observable: the
constant, the schema, the walk of the graphs. Never the sentence about it.

**Prose carries pointers. Records carry numbers.** A measurement in a sentence
is a copy that cannot say what it was measured against, on which commit, at
what canvas, in what cache state. The sentence keeps what was compared and the
direction of the result, and points at the script, the dated record, or the
constant that owns the value. A number with no findable origin gets no
invented pointer: delete it or mark it unsupported, in place.

**A number the reader does not act on is decorative.** Substitute a different
plausible value; if the next action is unchanged, delete it. Generating a
fresh value is not a lesser fix. It makes the claim permanently true and
permanently useless.

**Say what kind of evidence you have, inside the claim.** Measured, inherited,
reasoned, read from source, reported by someone else. A trailing hedge gets
trimmed; a label inside the sentence survives quotation.

**Withdraw out loud.** When a claim loses, correct it and say what it used to
claim. A reader who remembers the old sentence needs to see it retracted, not
quietly reworded.

## Measurement

**A default is not a decision, and shipping is not evidence.** A value that
sits in every graph has standing through repetition. Before citing one, ask
what it would take to find out it was wrong, and say so if the answer is a
record nobody has written.

**Measure where the model was trained.** A cheap canvas is for making a
harness run. A number that will inform a shipped decision is taken at a
trained canvas, because small canvases have inverted findings.

**Compare knobs at the call, not at the output.** A perturbed sampling
trajectory is a different sample, not a degraded one, so two rendered clips
cannot A/B a numerical change. Grade a kernel on captured activations against
an exact reference. A perceptual claim needs a distribution of seeds judged
blind, never a pair.

**Capture broadly first; decide what it means second.** A probe that varies
one axis has assumed the answer is on that axis, and a run that records one
number per arm cannot be re-asked. Spend the design effort on what to record.

**"It works" is not "it was trained for".** A capability that functions
because a code path has no bounds check is not one the model learned. The
question of any input is whether the vendor's own pipeline ever produces it.

**The second run is not the first measurement.** Caches, resident models and
warm weights make a timing a statement about cache state as much as about
configuration. Say which state you measured in.

## Controls

**A check that cannot fail is not a check.** Ask what its input would have to
look like for it to go red. Prefer a control it compares against, from an
independent implementation, over numbers the check computed itself.

**A requirement is not a control.** When a document says "must", name the
assertion that goes red if it is ignored, or write "enforced by nothing". The
requirements most likely to lack a control are the ones everyone agrees with.

**No new check until an instance escapes the existing ones.** The reflex to
add an instrument is usually a failure to read the ones installed. Cite the
escaped instance before building.

**A red result on a correct state is worse than no check.** It trains the
reader to ignore red. An allowlist that carries its judgement is better than
a grep that cries.

**Building the replacement is not the change.** Retiring the original and
repointing everything that cites it is the change.

**A search that returns nothing has established nothing** unless you know it
could have matched. An empty result is evidence about the pattern. Derive the
set and inspect it, or read the section.

**Re-reading your own work does not meet the standard.** Defects here have
been found by a second reader, nearly without exception.

## Structure

**One source, no second copy.** A shared constant lives in one place.
Anything that can be generated from it is generated, never retyped, and the
generated file is never hand-edited. Editing the generator is half the change;
nothing is true of the output until it is rebuilt.

**A numeric input means the quantity it names.** A mode gets its own named
input, never a magic value of a number.

**Record provenance on agreement.** Two things that share an owner agreeing is
consistency, not corroboration, and the artifact cannot tell which it was
later. Write down whether an agreement was borrowed or reached separately.

**Correctly absent is not broken.** When something gains an off, parked or
absent state, every assertion about it inherits a third case.
