# Red harnesses: why each one exists

A red harness proves a check can fail. `harness.py` explains the mechanism and
`spine_control.py` is the control on it. This file is the other half: the
instances that made the mechanism necessary.

**The operative rule is in `CLAUDE.md`** — a check whose input already satisfies
the expected outcome cannot fail. This file does not restate it. It records the
four separate times this repo found that rule violated in code, because the
generalisation is easy to nod at and the instances are what make the cost legible.

Moved here 2026-08-17 from the postmortem collection. They sat among frozen
incident narrative, where nobody writing a harness would read them; they belong
beside the fixture that exists because of them.

---


6. **`sol_curves.verify_adjacency` was, until 2026-08-16, a check whose input
   could not fail** -- the clearest instance in this repo of the standard above
   being violated in code rather than in prose. It counts non-adjacent steps
   along the Hilbert curve, and its only call site
   (`bench/analyze_capture.py`) passed `side=64`. On a power-of-two square,
   adjacency is Hilbert's *defining property*: zero is what every correct
   implementation returns, so the assertion could only ever go red on a
   corrupted `hilbert_d`, never on the ordering being scored. Meanwhile the
   ordering actually applied is a 24x42 rectangle clipped out of that square,
   which splices the curve in **6 places of 1007**. So the repo asserted "a
   Hilbert curve never jumps" in `sol_curves.py`, in `docs/morton.md` and in
   the node's UI tooltip, and held a green check that was structurally incapable
   of contradicting it.

   Fixed by giving `verify_adjacency` `height`/`width` parameters and calling
   both forms: the square stays a **gate** (non-zero means `hilbert_d` is
   broken), the rectangle is **reported, never gated** -- a non-zero result
   there is expected, and no threshold that would make it pass/fail has been
   established. Asking "what would the input have to look like for this to
   fail?" is the question that finds this class, and it is worth asking of every
   row in the index above.

   **A control cannot see a branch its input never reaches, and a
   before/after snapshot is exactly that shape.** Third instance of the family,
   2026-08-16. When `_ref_prompt` gained per-socket roles, the control was a
   byte-identity snapshot: all 43 prompts captured *before* the change, 0 of 43
   different after, 0 of 87 regenerated graphs changed. That is a real control
   and it proved what it claimed — the additive constraint held **for the path
   that already had graphs**. It could not have caught anything on the new
   path, because no graph exercised the new path until one existed. Two defects
   were sitting there and both surfaced within ninety seconds of the first
   three-role graph being built:

   - the generator defined a garment subject, gave it `attribute_transfer`, and
     never cited it in `detailed_description`;
   - `check_ref_prompt_labels.py` enumerated `images` as `(True, False)`, which
     cannot express a role tuple, so the first honest three-role graph was
     reported as a hand-edit. Its own comment said a hardcoded copy "stops
     covering the generator the moment a role is added" — it was right about
     the risk and still missed it, because **what changed was the *shape* of
     the argument rather than one of its values, and an enumeration written
     over old values is blind to a new form.**

   Same family as `verify_adjacency` (input cannot fail) and the identity
   control that compared `arange` against `arange` (input is the expected
   answer), one level up: **the input never reaches the code under test.** The
   question that finds all three is the same one — *what would the input have
   to be for this to fail?* — and for a regression snapshot the answer is
   "something that did not exist when the snapshot was taken."

   The fix generalises too, and it is cheap: **when adding a code path, add the
   first caller in the same change and let the existing checks judge it.** A
   new branch with no caller is unobserved by construction, however green the
   suite is.


---

## Where the numbering came from

The items below kept the numbers they carried in `docs/checks.md`'s Gaps list,
where they were items 6 and 7. That list has since been compressed to one line
per gap, so the numbers no longer resolve anywhere -- kept only because the
surrounding prose refers to them.

---

## Before adding a fourth harness

Tier 1 is deliberately small: a committed harness here is for an invariant whose
violation is **silent and external** — it breaks something no check in this repo
can see, like the owner's saved graphs outside it. `node_id` qualifies. Most
things do not.

**A fourth harness requires a drift instance that a filed audit run provably
would not have caught.** That is the specific form of `CLAUDE.md`'s no-new-check
rule, and it exists because the alternative was going to be thirteen harnesses
for thirteen index rows that claim calibration with no artifact. Those are tier
2: `test-audit`'s step 4 is oracle verification by spot mutation, dispatching to
`adversarial-verify` with a separate needle pass, and it is the shipped
instrument for exactly that job.

If a harness does earn its place, put it on the spine. Do not copy an existing
one — the three that predated `harness.py` were copy-paste of each other, and the
copies diverged until one had no expected-outcome comparison at all.
