# Tier 1: what must be true before the observer gets a render

**Opened 2026-08-31, with the instrument built, green and deliberately
unstarted** at the owner's call. Nothing here has spent card time.

This file is **only the gate**. What was measured on the weight side, the four
review defects that reproduced against the observer, and the method lessons the
day earned are [`2026-08-31_handoff.md`](2026-08-31_handoff.md)'s and are not
restated. The experiment itself is
[`../../open_experiments.md`](../../open_experiments.md) #23; the lever
inventory is [`../quant_levers.md`](../quant_levers.md); the joint-capture
contract is `internal/codex/2026-08-31_joint-tier1-capture-and-quant-reachability-review.md`.

**Why a gate at all.** The instrument passed eight acceptance cases and still
carried a defect that would have allocated 2.79 GiB per MLP on a 24 GiB card.
The cases were green because they ran 64 rows. A capture that fails at
production geometry costs a render; one that succeeds while recording the wrong
step costs a render *and* produces a file nobody can tell is wrong.

---

## The nine items

None of them needs the card. All of them are why the render has not run.

### 1. A dedicated bench/capture graph, and only that

The observer must not enter the canonical workflows. Instrumentation stays
environment-gated and hard to arm by accident, which is the same rule
`h3_capture.py` states about itself. Generator work in `build_workflows.py`, and
**nothing is true of a graph until it is rebuilt**.

### 2. `H3_QUANT_OBSERVE` in `bench/restart_comfy.sh::ARMING_KEYS`

The server process is the resource, not the GPU. An armed server nobody can see
is what cost three renders on 2026-08-30, one of them killed at 345 frames.
`ARMING_KEYS` is one list to extend and the guard already exists.

### 3. A disk-budget print before execution, not after

    Q/K/V, 20 cells                    ~84 GiB per render
    all four linear inputs             ~6.27 GiB per (block, step)
    naive all-modules full inputs      ~2.45 TiB        <- not the design
    quant sidecar                      negligible

Print the expectation at arm time. Discovering it is how a batch gets thrown
away.

### 4. A full-input allowlist keyed by `(block, step, kind)`

It must **refuse** an unbounded request at production geometry rather than
honour it. The first exact decomposition wants two pre-registered cells — the
control kind and the suspected outlier-heavy kind, at middle and deep depth.

### 5. The join to `H3_CAPTURE` is open

The observer generates a `capture_id` at install; `h3_capture.py` does not know
it. Close this **before** the render, by either publishing one id to both
instruments or documenting the join on `(block, step)` plus render identity —
and say which. Two instruments inferring indices independently is how two
plausible files end up with different counters, which is the failure the
contract's "one shared authority" clause exists to prevent.

### 6. `bench/grade_quant_on_capture.py` does not exist

It is the offline scorer and it is where the answer actually comes from. The
capture is worthless without it, and writing it after the render is how a
capture turns out to be missing the one field the scorer needed.

### 7. The four arms, defined before they are coded

| arm | weight | activation | measures |
|---|---|---|---|
| reference | released BF16 `W` | BF16 `X` | the denominator |
| runtime W8A8 | shipped `Q(W)` | rotated, row-quantised `X` | the actual combined error |
| weight-only | dequantised `Q(W)` | BF16 `X` | what every existing record measures |
| activation-only | BF16 `W` **in the matching rotated basis** | rotated, row-quantised `X` | the term nothing has measured |

**The activation-only arm is the subtle one.** It is not "exact weight with
quantised activation": the runtime activation is rotated before it is
quantised, so an unrotated reference weight makes the arms incomparable. An
earlier draft of #23 said the simpler wrong thing.

And the rule that came out of `full_precision_matrix_mult`: **do not name a
flag as an arm until its output has been checked against the arithmetic it is
meant to implement.** That one was entered in the lever inventory from a source
trace and an executable probe found it changes nothing at all.

### 8. A composition test against a real model

The stub proves the observer chains rather than evicts. It does not prove
observer + PDD + Sol in one graph. Assert **output bit-identity with the node
wired against bypassed**, on a short run, before anything is quoted — an
observer that changes the tensor makes every number it records a statement
about a different model.

### 9. The canvas, and the second scene

Production is **1344x768, 345 frames**, per the owner: sequence composition is
the variable being measured, so a cheap canvas would change it. Run a throwaway
small-canvas pass first to exercise the harness end to end and do not quote its
numbers.

**A joint render replaces a duplicate render of the SAME scene. It does not
replace the second scene/seed that generalisation needs.**

---

### 10. Choose the cells from the live route record, not from a guess

Added 2026-09-01. The Sol route record (`sol_observe.py`, armed by
`H3_SOL_OBSERVE`) now reports, for every attention call of a real render at
the trained canvas, which route it took and how dense the routed walk was,
by block and step -- `bench/results/2026-09-01_sol_route_pdd8_cold.json` is
the 8-evaluation PDD graph as shipped. That is a few hundred kilobytes per
render against the tens of gigabytes a Tier 1 activation capture writes, so
it should decide WHICH `(block, step)` cells the capture holds: the
in-window forwards are knots 8 through 20 on the 8-evaluation grid (four of
eight), the dense blocks are 0-2 and 32, and everything outside the window
runs on Sage through the composed patch. A cell the router never touches
needs no Sol-side capture; a cell with an unusual density is the one to
spend disk on. Item 8's bit-identity assertion still applies to the
observer + PDD + Sol composition; the route record has its own
(`bench/check_sol_observe.py`, `bench/grade_sol_record.py`).

## What a good first capture looks like

- `shape_check.ok` true, four kinds at every requested block and step, no
  `failures` and no `incomplete`.
- `segments` present and non-null — a capture that binned positionally has lost
  the grouping, which is what leaves `grade_sage_on_capture.py` unable to answer
  a segment question today.
- Steps distinct and monotone against `sample_sigmas`, not all equal. All-equal
  is the signature of the scale bug and it looks complete.
- Per-module peak in the 190-230 MiB band measured on 2026-08-31; materially
  above that means a chunking path was bypassed.
- The disk budget printed at arm time matching what landed.

## What it cannot tell you, however clean

It is the **activation** half. It is not a runtime error until the scorer runs,
not a perceptual claim under any circumstances, and its **timing is void** —
reductions and CPU copies do not change values but do change wall time and peak
pressure. The `convrot_groupsize` timing arm that #23 gates needs its own
uninstrumented process.
