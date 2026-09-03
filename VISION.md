# What this repo holds itself to

This is a research hub for running and studying MiniMax H3 inside ComfyUI on
one box. Its output is not the nodes or the workflows. Its output is claims
about the model that a later reader can re-derive, and the tooling that makes
re-deriving cheaper than trusting.

This file is the why. It should rarely change. `CLAUDE.md` is the operative
form, one line per rule; `docs/rules_history.md` is the dated instances that
earned them; `docs/` carries the stories. If a tenet here turns out to be
wrong, change it in a commit that says what it used to claim and what refuted
it.

## Truth

**Prose is a cache. The observable is the source.** Code is right when prose
disagrees. A measurement belongs in the script, record or constant that owns
it, and the sentence keeps only the direction and a pointer. A number the
reader would not act on differently if it changed is decorative: delete it
rather than refresh it.

**Label the evidence, and withdraw out loud.** Say inside the claim whether it
was measured, inherited, reasoned, read from source, or reported by someone
else. When a claim loses, correct it and say what it used to claim, so the
reader who remembers the old sentence sees it retracted rather than reworded.

## Measurement

**Nothing has standing by being there.** A default, a shipped value, a
capability that happens to work because a code path has no bounds check. Ask
what it would take to find out it was wrong, and whether the vendor's own
pipeline ever produces this input.

**A measurement is a statement about its conditions.** Which canvas (a trained
one, or it can invert), which cache state, which commit. Two rendered clips
cannot A/B a numerical knob, because a perturbed trajectory is a different
sample: compare at the call, on captured activations. A perceptual claim needs
a distribution of seeds judged blind, never a pair.

**Capture broadly first; decide what it means second.** A probe that varies
one axis has assumed the answer is on that axis, and a run that records one
number per arm cannot be re-asked. Spend the design effort on what to record.

## Controls

**A control must be able to go red, and only when the state is wrong.** A
check that cannot fail is not a check. A "must" with no assertion behind it
says "enforced by nothing". A check that is red on a correct state teaches the
reader to ignore red, which is worse than no check.

**Add nothing until something escapes; retire what you replace.** The reflex
to add an instrument is usually a failure to read the ones installed. Building
the replacement is not the change; retiring the original and repointing what
cites it is.

**Your own reading is not verification.** An empty search is evidence about
the pattern, not the corpus. Re-reading your own work has caught almost none
of the defects found here; a second reader has. Two things that share an owner
agreeing is consistency, not corroboration, so record on the agreement whether
it was borrowed or reached separately.

## Structure

**One source, generated copies, never hand-edited.** A shared constant lives
in one place; anything derivable from it is generated, and editing the
generator is half the change because nothing is true of the output until it is
rebuilt. A one-off experiment may skip this, if you are certain it is a
one-off. A numeric input means the quantity it names, and a mode gets its own
named input. When something gains an off, parked or absent state, every
assertion about it inherits a third case: correctly absent is not broken.
