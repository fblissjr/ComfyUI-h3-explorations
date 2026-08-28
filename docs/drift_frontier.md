# Drift work: the open frontier

Tracking file for the doc-drift and control-calibration work opened 2026-08-17.
One row per decision. **Annotate rather than rewrite** — when an item resolves,
change its status and add a dated line under it saying what was decided and why.
A row whose reasoning is edited away stops being a record.

Status values: `open` (needs the owner) · `settled` (decided, not built) ·
`done` (built and verified) · `dropped` (decided against, reason recorded).

**Next sweep: 2026-09-28.** At a sweep every `open` row is either moved to
`settled`, `done` or `dropped` -- `dropped` with a recorded reason is a result,
not a failure -- or re-dated with why it is still open. A row that survives two
sweeps unchanged is `dropped` in everything but name and should be relabelled.
Adopted 2026-08-28; [`sustainability.md`](sustainability.md) argues the case.

Numbers here are descriptive, so they carry an observation point. Anything
without one is a defect in this file.

---

## Done

### D1 — Red-harness spine, its control, and three ports
**Done 2026-08-17.** `bench/red/` now holds `harness.py`, two control fixtures,
`spine_control.py`, and the three harnesses ported onto it.

The defect: all three harnesses computed an expected-outcome mismatch and then
exited 0 unconditionally, so the recorded evidence for three rows of
`docs/checks.md` was a program that returned success whatever happened. One of
the three had no comparison at all.

The design that avoids a regress: a case carries a KIND, not an expected
verdict. `MUTATION` asserts the verdict differs from baseline, `NEAR_MISS`
asserts it matches. One derived rule for every case, forever, and it doubles as
the needle check — a mutation that never reached its subject leaves the verdict
unchanged, which is what `MUTATION` already asserts. An exception is `ERROR`,
never "differed".

The spine has its own control because shared infrastructure fails silently
across every harness at once. `spine_control.py` runs an inert-mutation fixture
that must exit 1 and a healthy fixture that must exit 0 — red for the right
reason and green for the right reason, both structural rather than authored.

Verified 2026-08-17: `spine_control.py` exit 0 (2/2); node_ids 8/8 exit 0;
graph_discovery 8/8 exit 0; analyze_routing exit 2 with its reason when
`H3_CAPTURE` is unset. No absolute or home-directory paths in `bench/red/`.

Two things fixed in passing: the capture location is now read from the
`H3_CAPTURE` environment variable rather than a home-directory literal (a leak
by `path-privacy`'s rule, and it named a machine), and the graph-discovery case
filenames are index-based rather than `abs(hash(label))`, which changed per
interpreter run.

**Annotation, 2026-08-17: this was marked `done` before it was.** The ports
landed but the pre-port copies stayed in `scripts/experimental/`, differing from
them, and the old `show_red_check_node_ids.py` still exited 0 when run. Worse,
`docs/checks.md` went on citing them — one by full path to the superseded file,
two by bare basename, which had become ambiguous between two locations and which
`check_doc_links.py` does not sweep. So the calibration column cited programs
that return success no matter what happens, which is the defect this entry
claims to have fixed.

Closed properly the same day: the three pre-port copies moved to
`scripts/archive/sol_curve_2026-08-16/` with a README saying not to run them, and
the three citations rewritten as `bench/red/<harness>.py::build` — the
`path::symbol` form, so `check_doc_links.py` now goes red if a harness is moved
or its entry point renamed. The `#18` row's caveat about an uncalibrated harness
was removed, since it had become false.

The lesson is the one this file keeps recording: building the replacement is not
the change. Retiring the original and repointing what cites it is the change.
Status: `done`

---

## Open — needs the owner

### F1 — Where the tier-1 / tier-2 line sits
Tier 1 = a committed re-runnable harness in `bench/red/`. Tier 2 = calibrated by
a filed audit run.
- (a) the three that exist, plus anything guarding an invariant whose violation
  is **silent and external**
- (b) zero: delete the calibration column, go fully audit-time
- (c) all sixteen rows claiming shown-red

**Recommended: (a).** Peer concurs; the silent-and-external criterion is the
part that transfers to other repos. Unlocks the most — under (b), F3 and F4
disappear and F5 shrinks.
Status: `open`

### F2 — The cells claiming a mutation with no artifact
Most rows claiming `shown red` have no runnable artifact behind them; three do,
in `bench/red/`. **Derive the rest rather than reading a number here** — the
index table's own column is the authority, and the counts in this entry drifted
within hours of being written: `bench/check_*.py` went from 23 to 24 the same
afternoon when another agent added one. That is the failure this file exists to
track, committed by this file.
- (a) they become tier 2: a **filed**, dated audit run is the evidence
- (b) relabel "recorded, not runnable" and leave them
- (c) convert to harnesses over time

**Recommended: (a).** `test-audit`'s step 4 is oracle verification by spot
mutation, dispatching to `adversarial-verify` with a separate needle pass — the
shipped instrument for exactly this. The run must be **filed**, not just
performed: an unfiled run makes the cell an unsourceable status claim, which is
transcribed claims replacing transcribed claims.
Status: `open`

### F3 — The calibration column's shape
Two values, both derived: a `path::symbol` citation to a harness in
`bench/red/`, or a citation to a filed calibration run.
**Recommended: adopt both.** Stops the column being the standing artifact
`control-audit` explicitly warns against.
Status: `open` (depends on F1)

### F4 — How a tier-2 cell cites its run, given a private collection
F7 keeps postmortems gitignored; the cell lives in a tracked `docs/checks.md`.
That is `filing.md`'s unreachable-citation case.
- (a) the cell carries the load-bearing finding inline
- (b) calibration runs file tracked while incidents stay private — splits the
  collection, which F7 says not to do
- (c) the column stops living in a tracked file

**Recommended: (a).** Tripwire: if a finding will not compress to a clause, the
cell is carrying more than a cell should, and that record wants to be a
document. A set of those would mean tier 2 is the wrong shape.
Status: `open` (depends on F1, F7)

### F5 — Citation rewrite scope
`path::symbol` is toolchain doctrine (`adversarial-verify` 0.9.0: "the checkable
form: it survives insertions above"). Scope, smallest first:
- (a) the requirements table plus the tier-1 calibration cells
- (b) plus the index table, which is much the larger set
- (c) every bare `.py` basename in `docs/`

**Precedent exists.** Three cells were converted on 2026-08-17 as part of D1 and
shown red by renaming an entry point, so the form is proven here rather than
imported. Derive the remaining scope with a grep for backticked basenames
carrying no directory and no `::`; a count typed here would be stale by the next
commit, and the first draft of this entry was.

**Recommended: (a).** Prove the pattern on the small set before the large one.
Status: `open`

### F6 — Regex gate on citation form
"Does this cell contain a backticked `.py` without `::`" — pure syntax, no
judgment, unlike the refuted enforcement-detection check.
**Recommended: yes, last**, after F5 has run once. A gate on a rewrite that
never repeats guards nothing.
Status: `open` (depends on F5)

### F7 — Postmortem collection: tracked or private
`filing.md` 0.9.0's unreachable-citation rule means the gitignored-path
citations are not defects while the collection stays private; they would be
created by moving. Privacy scan 2026-08-17: no identity, credential, absolute
path, email, IP or key-shaped content in any of the 11 files, independently
re-derived with a positive control. Two third-party items block publication, one
of which needs a maintainer contacted first.
**Recommended: private.** `.postmortem.json` to `internal/postmortems/`; the
outlier created 2026-08-17 in `docs/` moves there and gains frontmatter. Decide
for the collection, not per file — `postmortem-index` resolves one directory, so
a split shows partial history as though complete.
Status: `open`

### F8 — Closure-back pass over the existing postmortems
`filing.md` 0.9.0: a body is frozen but forward items "do not inherit the
freeze". One live candidate is known, in a gitignored postmortem this tracked
file therefore cannot cite by path — its deviations table carries a row reading
"Relay four items to [a named maintainer] ... | Recorded, none relayed |
Deferred", which is a forward item with no closure annotation.
**Recommended: yes**, bounded to `filing.md`'s stated scope. Applies either way
on F7.

*The edit above is itself F4 option (a) in practice: the pointer was a bare
basename into `internal/`, `check_doc_links.py` went red on it, and the fix was
to carry the load-bearing sentence inline and drop the path.*
Status: `open`

### F9 — Delete the stale run-log lines from `docs/check_postmortems.md`
Their only defence was that "did not apply" and "did not run" are different
states; the next full run supersedes them, and their counts contradicted the
table in the same file.
**Recommended: delete.** Ordinary editing under the corrected deletion rule; git
history keeps them.

**Done 2026-08-17, phase 1.** The `## Run history` section is deleted -- its
counts had drifted three ways and contradicted the table in its own file, and
the next full run supersedes it. Git history keeps it. The file's header used to
advertise the run logs and the four control instances; both claims were stale the
moment the content left, and the header was corrected in the same edit.
Status: `done`

### F10 — Where cross-cutting lessons live
**Resolved 2026-08-17 to "no new surface", pending the owner's yes.** The
material is two principles, and both are already stated in `CLAUDE.md` — "a
check whose input already satisfies the expected outcome cannot fail", and "say
which kind of evidence you have inside the claim". What was going to be
relocated is their *instance narrative*, not the principles. Instances for the
first belong in `bench/red/`'s own docs, next to the fixture that exists because
of them; the second stays in the postmortem.

A design-records surface was also declined: each of this session's decisions has
a home in the thing it governs (the tier line in `checks.md`'s standard, the
citation form beside the column, the postmortem location in `.postmortem.json`,
the spine in `bench/red/`). The one candidate lacking a home — the
private-versus-tracked decision and its premises — is a single instance, so the
tripwire says defer.

**Done 2026-08-17, phase 1.** No new document. The four instances moved to
`bench/red/README.md`, beside the spine and fixture that exist because of them.
Both principles were already stated in `CLAUDE.md`, so only the instance
narrative needed a home, and the design-records surface stayed declined.
Status: `done` (confirm the "no new surface" call)

### F11 — The number rule in `CLAUDE.md`
**Reversed twice; current state verified 2026-08-17.** `dev-conventions` 0.17.0
ships zero hook files and the `claims-in-docs` directive is gone, so the earlier
plan — delete the local rule and let a directive carry it — would have left no
statement of the rule anywhere. It survives only in
`/dev-conventions:doc-conventions` and `claim-audit`, both invoke-only.
**Recommended: keep it local and upgrade the phrasing** to the three-part form:
substitution test; normative versus descriptive; binding is not a lesser fix for
a decorative number. No change needed to `docs/checks.md` — the ground regex it
matched no longer exists.

**Done 2026-08-17, phase 1.** `CLAUDE.md`'s Guiding Principles now carry the
three-part rule: substitution test, normative versus descriptive, and generating
a decorative number is not a lesser fix than deleting it. It points at
`claim-audit` for prose already written. No `docs/checks.md` change was needed.
Status: `done`

### F12 — Record this work in `CHANGELOG.md`
The doc restructuring landed inside `d21e504`, a commit whose message is about
prompt templates, and `CHANGELOG.md` had not been touched since `ce41e3f`.
**Recommended: one entry** describing what actually landed.

**Done 2026-08-17.** A `0.34.0` section covers the spine and its control, the
harness ports, the two false requirement rows, the absolute paths, the moved
attributions, and both document restructurings. Minor bump: additions and fixes,
no removals from a published surface. The entry describes the work rather than
citing `d21e504` by hash — the invisibility being fixed was that nothing
described what landed, and a hash does not describe it.
Status: `done`

### F13 — Promote `probe_hilbert.py` to `bench/`
Five siblings import it, and `bench/analyze_canvas_geometry.py` cites it as the
independent second implementation its non-tautology claim rests on.
**Recommended: yes.** Archiving it would silently void that claim.
Status: `open`

### F14 — The tripwire
General: no new drift check until a drift instance appears that the existing
gates provably could not have caught.
Specific: a fourth tier-1 harness requires a drift instance a filed audit run
provably would not have caught.
**Recommended: both.** Peer concurs that the specific form is the falsifiable
one. It binds the agent more than the owner, which is the point.

**Done 2026-08-17, phase 1.** The general form is in `CLAUDE.md`'s rules, with
the day's six dissolved proposals as its evidence. The specific form -- a fourth
tier-1 harness needs a drift instance a filed audit run would not have caught --
is in `bench/red/README.md`, next to the tier-1 criterion it bounds. A third rule
went in beside it: building the replacement is not the change.
Status: `done`

### F15 — Disambiguate the int8s
Three distinct ones are in play and no document says which a claim means:

- **the DiT and CLIP checkpoints** — `workflows/h3_config.py` names all three
  `int8_convrot`, and `README.md` lists them in the shipped config
- **the Sol-Attn CUDA kernel** — `docs/bench_plan.md` says it "routes in INT8
  unconditionally", with `cuda-int8` in the log as the tell
- **the VAE decoder** — a separate `int8_convrot` build, noted in
  `workflows/h3_config.py`

The cost is live rather than hypothetical. `docs/SOLATTN.md`'s Morton
retraction gives "stacked on int8" as its stated reason, and which int8 that
means changes what follows from it. The attention kernel is the stronger reading
given the surrounding Triton and tau context, but **that is an inference from
context, not something the row states.**

**Revised 2026-08-17** after the owner pointed out that the property is not just
int8 but **pruned, convrot, int8** — three separate things — and that it can
condition anything, not only Sol. That splits the fix across two homes:

1. **The three int8 layers → `docs/SOLATTN.md`.** It owns every Sol number and is
   where "stacked on int8" is quoted, so the disambiguation belongs beside it.
2. **The substrate itself → `docs/evidence.md`'s "Environment, because it is not
   in git".** That section exists for exactly this: un-versioned things a
   measurement depends on. It records the `comfy-kitchen` build, `nanobind`, the
   vendor symlink and the coderef clone, and closes with "record the node hash
   and the `comfy-kitchen` tag with any measurement" — **and it never names the
   model build.** Every number in this repo was measured on pruned,
   convrot-rotated, int8 weights, and `fp8_scaled` and `w4a8_mixed` builds of the
   same models sit on disk beside them, so nothing about a recorded measurement
   says which one produced it.

The original recommendation was `SOLATTN.md` alone, which would have filed a
global condition on every measurement inside a Sol-specific page.

**Half done 2026-08-17, phase 2.** The three-int8 table is in
`docs/SOLATTN.md`, under "Read this before quoting any number on this page" and
ahead of the Do-not-rely-on table, because it conditions how every row there is
read. It names which layer each sense refers to, how to tell them apart, and that
the Morton retraction means the attention kernel — a reading taken from context
rather than from the row, which is stated as such. It also carries the standing
condition: pruned, convrot-rotated, int8, with `fp8_scaled` and `w4a8_mixed`
builds on disk beside them.

**Done 2026-08-17, phase 2.** The `docs/evidence.md` half landed in the
Environment section, after the peer session finished with that file and proposed
the split: `docs/hardware.md` owns the physical machine, the Environment section
keeps the software substrate, and the model build is software. It records the
three properties, why convrot's rotation is not just bookkeeping, and where a run
does and does not capture which build produced it.
Status: `done`

### F16 — Does convrot's rotation reach Sol's routing or Morton's ordering?
`int8_convrot` stores `W @ H^T` in a Hadamard basis rather than the weight
itself (measured 2026-08-21, `bench/analyze_quant_delta.py`), so anything read
out of those files without `dequantize_int8_convrot_weight` is in a rotated
basis.

Sol reorders and routes at a **64-token block**, which is an 8x8 tile in 2d and
4x4x4 in 3d. The reordering's whole premise is that tokens adjacent in that tile
carry similar q/k values, so grouping them yields a tight centroid and the block
becomes skippable at a given tau. **Three properties of the shipped weights
attack that premise independently:**

- **int8** adds quantization noise to q/k, inflating within-block variance
  whatever the ordering — so the locality signal competes with noise the ordering
  cannot reduce.
- **convrot** applies a rotation, so "similar in value" is judged in a rotated
  basis. Whether spatial adjacency still maps to proximity there is not
  established anywhere.
- **pruned** means the value distribution being exploited is the pruned model's.
  A locality result need not transfer to unpruned weights, and pruning may
  itself have removed the structure that carried spatial coherence.

**Evidence kind: inferences from source reads, not measurements.** None of the
three has been tested against the ordering. Together they are the mechanism that
would explain a Morton gain evaporating on an int8 path, which is the claim that
was retracted.

The control is better than first written: `fp8_scaled` exists as a **matched
build of both models**, not merely as some other quantization the repo happens to
hold. So `int8_convrot` against `fp8_scaled` within one model role is a
one-variable comparison, and no new render is needed. A `w4a8_mixed` fl2va build
gives a third point if the first two separate.

**Blocker: capture provenance.** `docs/capture_manifest_schema.md` carries a
`weight_quantization` field, but `workflows/h3_config.py` records that the loader
"prints `dtype: torch.float16` for both builds and cannot distinguish int8
storage from a dequantized fallback" — so which build produced a capture is
unrecoverable after the fact unless the manifest recorded it at capture time.
Whether existing captures populate that field is **unchecked**; they live outside
the repo, so this file cannot answer it.

**Recommended: file as a named unrun item in `docs/open_experiments.md`**, whose
format is exactly what is deliberately not measured and the blocker for each. Not
a new document, and not a claim in `SOLATTN.md` until it is measured.

**Done 2026-08-17, phase 2.** Filed as open experiment 19 in
`docs/open_experiments.md`, in that file's own format: what it tests, the three
properties as separate attacks on one premise, the evidence kind stated inside
the entry, the method, and the blocker.

Two things sharpened in the writing. The control is a **matched** `fp8_scaled`
build of both models, so the comparison is one-variable within a role rather than
confounding quantization with model role. And what to compare is the routed
density and the centroid variance the router actually reads, not wall-clock —
wall-clock is what produced the retracted claim in the first place.

**Correction, 2026-08-17, same day.** The blocker as first filed was wrong and
a peer session caught it by checking the schema instead of reading my summary.
`models` requires `unet`, `clip` and `video_vae`, and those filenames are
self-describing, so a conforming capture manifest **does** say which build ran.
The real gap is tighter: `weight_quantization` and `vae_quantization` are in no
`required` list and `bench/check_capture_manifest.py` inspects neither, so a
manifest can omit both and pass green — and a bench run emits no manifest at all,
so a timing records nothing about its weights. Corrected in
`docs/open_experiments.md` and in `docs/evidence.md`. This is the second time
today a claim of mine survived my own review and failed a second reader's.
Status: `done`

### F17 — An index-inventory check, which now clears the tripwire
**Two instances, different authors, hours apart.** `check_graph_discovery.py` was
on disk with no row in `docs/checks.md`'s index, found 2026-08-17 morning. That
afternoon `check_capture_manifest.py` landed the same way, added by the other
agent working in this tree. Neither was caught by anything.

This is the check I proposed in the morning and then dropped, correctly, for
having no evidence. **F14's tripwire now reads satisfied:** a drift instance
appeared that the existing gates provably could not catch, twice, because
`check_doc_links.py` verifies that citations resolve and says nothing about
whether a file that exists is cited at all.

The assertion is mechanical with no judgment in it: every `bench/check_*.py` has
a row, and every row names a file that exists. It is the inverse direction from
the check that was refuted this morning — that one tried to detect *enforcement*,
which is semantic; this one compares two directory listings.

**Recommended: build it, after F5**, so the rows it asserts against are already
in the citable form. Give it a red harness in `bench/red/` on the spine — adding
a check without a row is the mutation, and it is trivially constructible.

**Partly done 2026-08-17, phase 3.** Two concrete pieces landed once the peer
released `docs/checks.md`:

- `bench/check_capture_manifest.py` now has an index row, so every
  `bench/check_*.py` on disk is indexed — verified by comparing the two listings,
  which is the assertion this entry proposes to automate.
- **A real defect fixed in it.** The accepted-version set and the reported version
  were separate literals and had already drifted: the report printed a hardcoded
  `v1.0.0` while the schema and the only existing manifest were both `1.1.0`, so it
  would have kept claiming `1.0.0` through every future bump. Now one
  `SCHEMA_VERSIONS` constant gates acceptance, and the report states the versions it
  actually saw. Verified: it prints `v1.1.0`.

**The check itself is still to write**, with its red harness on the spine — adding
a check without a row is the mutation, and it is trivially constructible.

**Done 2026-08-17, phase 3.** `bench/check_doc_inventory.py` plus
`bench/red/show_red_check_doc_inventory.py::build` on the spine. Both directions
asserted: every `bench/check_*.py` has a row, every `.py` a row names exists.

**It caught itself on its first run** -- on disk, no row -- which is the shortest
possible demonstration that it can fail.

**And its own harness caught a false red in it before it ever ran on the real
file.** Case G2 feeds prose that merely begins with a pipe; the first version read
any pipe-led line inside the section as a row, found no `.py` subject, and reported
an error. A false red, which this repo rates worse than no check. Fixed by
requiring five cells, which leaves M5 -- a genuine five-cell row with no `.py`
subject -- still red. The near-miss half of a harness is the half that earns it:
10 of 10 cases behave, and one of the four greens was load-bearing.
Status: `done`

### F18 — Make the manifest quantization fields required, and assert them
**Found 2026-08-17 by a peer session**, checking the schema rather than the
summary of it. `weight_quantization` and `vae_quantization` exist as properties of
`models` in `docs/capture_manifest_schema.md`, appear in **no** `required` list,
and `bench/check_capture_manifest.py` inspects neither — verified, the grep returns
zero. So a manifest can omit both and pass green.

Two halves, and the second is the one that matters:

- **captures** — make both fields required and assert them. Cheap, and the build
  is already recoverable from the required filenames, so this closes a gap in the
  record rather than in what is knowable.
- **bench runs** — they emit no manifest at all, so a timing carries no record of
  the weights it was measured on. That is where a number can be wrong and nobody
  can tell.

**This is probably one piece of work with the peer's power-state row**, which they
added to the Uncontrolled requirements table the same day: timings compared only
at equal host GPU power state, enforced by nothing, and unclosable without a
manifest field first and then an assertion. Same sequence, same missing artifact
for bench runs.

**Recommended: yes**, and sequence it — the field, then the assertion, then a red
harness on the spine for the assertion. Coordinate with the peer so the bench-run
manifest is designed once rather than twice.

**Field set, drafted 2026-08-17 with the peer session.** One design, because two
designs for "what substrate produced this number" would diverge, and the
power-state field is useless alone since bench runs are where both gaps bite.

Verified against the schema and the one existing manifest rather than assumed:

| field | status today | action |
|---|---|---|
| `workload.models.weight_quantization` | property exists, in no `required` list, never asserted | require, assert |
| `workload.models.vae_quantization` | same | require, assert |
| `provenance.gpu_power_limit_watts` | same — schema line 35, absent from provenance's `required` | require, assert |
| `provenance.gpu_power_limit_default_watts` | **now exists** as `host.gpus[].power_limit_default_watts`, along with min and max, added to `substrate.py` after this table was drafted | require via the substrate block |
| `provenance.gpu_clock_lock_graphics` | **does not exist** | add, require, `null` allowed |
| `provenance.gpu_clock_lock_memory` | **does not exist** | add, require, `null` allowed |

**The migration is free today and gets dearer per capture.** The schema gates on
`schema_version` (`bench/check_capture_manifest.py:54`, enum `1.0.0`/`1.1.0`), and the
single existing manifest already declares `1.1.0` and already populates both
quantization fields — `int8_convrot` for each. So requiring them turns nothing
red now. That argues for doing it ahead of the bench-run work rather than behind
it.

**Three states per field, not two — this is the constraint that matters.** A
missing key means *not recorded*; a present `null` means *confirmed absent*, as in
no clock lock was set. A validator that reads absence as "presumably stock"
rebuilds the exact hole, and it is `CLAUDE.md`'s rule that anything gaining an
absent state makes every assertion about it inherit a third case. Absence must
fail once the field is required, and `null` must pass.

**Why the shape of each new field.** The power limit is recorded *with* its
default rather than as a percentage: the ratio is what matters, the default
differs per card, so a bare wattage is uninterpretable elsewhere and a bare
percentage loses the absolute. Clock locks are recorded verbatim because `-lmc`
changes numerics and not merely timing — it overrides the driver's compute-state
memory margin, and GDDR6X faults on this generation surface as retries or wrong
bits rather than a crash, so a capture that fails its own checksum under a memory
lock would read as a kernel bug. Persistence mode gets no field: first-run latency
only, neither steady-state timing nor numerics.

`bench/hwinfo.py` already reads all of it from `nvidia-smi`, so the emitter is a
call rather than new plumbing. Do not write a second reader.

**The scope call is the owner's, not ours.** Bench runs emit no manifest at all,
so this is not "add a field" for them — it is "`bench_e2e_h3.py` must emit a
manifest". That is a real decision about what a bench run owes, and it is where
both gaps actually bite, since a timing is the number most often quoted and the
one carrying no substrate record. Raising it here rather than letting it be
discovered mid-implementation.

**Scope call answered 2026-08-17 by the owner, relayed through the peer session,
and it dissolves the options I had drafted.**

One record with a `kind` discriminator (`capture` | `render` | `bench`), a shared
substrate block, a per-kind payload, and required-ness conditional on kind. My
objection to a single schema was that a bench run has no tensors or token
accounting, so most of it would go optional and the capture variant would weaken
-- under a discriminator that does not happen, because the capture variant keeps
every requirement it has and the bench variant never mentions tensors. And my own
recommendation, a shared fragment plus two schemas plus two validators, was the
duplication I had just argued against at the field level, one level up. Two
schemas for "what substrate produced this number" diverge exactly as two field
sets would.

**Sequencing, and this order is deliberate.** Assert what already exists first --
`weight_quantization`, `vae_quantization`, `gpu_power_limit_watts` -- because the
migration is free today and dearer per capture. Unify second. Extend to bench and
stamp third. If the unification stalls the assertions have still landed; the
reverse is not true.

**The bench half is larger than "add a sidecar", and I had it wrong.**
`bench/bench_e2e_h3.py` persists **nothing** today -- verified, its only
`json.dumps` is an HTTP body, and the numbers go to stdout. So it is "make the
bench persist at all", including where the record lives and what the retention
story is. Records go beside the results rather than committed, because a committed
sidecar becomes a second home for numbers the docs also cite, and one-home-per-
number is the rule this repo keeps relearning.

**Retro-labelling needs nothing from us.** `docs/bench_plan.md` and
`docs/evidence.md` both carry it as of `c493895`, tightened in `46897f5`. Do not
add a third home.

**The seam:** the peer wires the emitter, this track takes the schema and the
assertions. The emitter returns the substrate block and nothing else -- no
knowledge of kinds, payloads or file layout, or it becomes a second place the
schema is encoded.

**Step one done 2026-08-17, `2736778` and follow-ups.** `weight_quantization` and
`gpu_power_limit_watts` required with presence-only assertions; harness on the
spine, 8 of 8. `vae_quantization` deliberately **not** required — singular over
two VAEs at different quantizations, which the ref graph proves and the existing
manifest demonstrates by recording a value true of one and false of the other.

**Two design rules came out of it, both worth more than the fields.**

*A projection must be checked as one.* `weight_quantization` is readable off the
required `unet` filename, so it is a second home for one fact. Kept, but the
checker now asserts the value appears in the filename, and the filename wins any
disagreement because it names the file actually loaded. Same test applied to
`vae_quantization` against `video_vae`, which also pins down what that ambiguous
field means. Precedent is `filing.md`'s rule that an `artifacts` list is a
projection of the citations and a disagreement means one is wrong.

*Substrate carries point-in-time facts; a payload carries what its kind can
measure over its own duration.* Settled on the clock-lock case: the locks are
unreadable on this driver, and both obvious answers were wrong — requiring the
field would let a guess be written as `null`, which the schema defines as a
confirmation, and recording one clock sample instead would put a decorative number
under a descriptive name. So the substrate block carries `unobservable` as a
first-class fourth state, and sampled clock behaviour is a bench-payload question.

Remaining: unify the record behind a `kind` discriminator, then extend to bench
and stamp.

**Step two drafted 2026-08-17.** The discriminated record is in
`docs/capture_manifest_schema.md`: one record, a `kind` of `capture`/`render`/`bench`,
a shared substrate block, a per-kind payload, required-ness conditional on kind.

Drafted against a **regenerated** key set rather than a pasted one --
`python substrate.py --keys <api-graph>` -- because a pasted list is a second copy
of the emitter's structure and drifts. Proven, not assumed: running it without a
graph warns and loses six leaf paths across the `graph` and `weights` subtrees, so
a promptless draft would have validated a shape nothing emits while missing two
whole subtrees, both sides green.

**The substrate requirements came out structural rather than enumerated.** Each
group carries a `state`, and only the group-level `state` keys are unconditionally
required -- they are the four-state discriminator, so their absence is the one thing
unrecoverable later. Subtrees are conditional: `present` requires the subtree,
`absent` requires nothing, `unobservable` requires `why` so the reason survives.
That turns the four-state rule from a convention into the shape of the schema.

`weights` requires nothing below `state`, deliberately: it returns less than a
manifest, cannot recover LoRA strength or rank, and infers quantization from
filename shape -- which the emitter states in the field name itself,
`quantization_inferred_from_filename`. Requiring anything under a shape we already
know is wrong would be the mistake `vae_quantization` just taught.

Also recorded: `docs/capture_manifest_schema.md` and
`bench/check_capture_manifest.py` are both named for one kind of a three-kind
record. The rename waits for step three rather than churning citations early, and
`check_doc_links.py` will go red on the move -- which is why those citations are in
`path::symbol` form.

Status: `open` (step three: the bench payload, behind the owner's persist call)

---

## Held — prerequisites still open

| item | waits on |
|---|---|
| Two publication blockers, redactions, inlining unreachable citations | F7 |
| Archive the superseded probes; repoint `docs/open_experiments.md` citations | F13 |
| Frontmatter/filename date mismatch in one postmortem | F7 |
| Whether the generator emits `::symbol` citations itself | F3, F5 |
| Do existing capture manifests populate `weight_quantization`? | a look at the capture store, which is outside this repo |
| Compare routing statistics across quantizations, `int8_convrot` against `fp8_scaled` | F16, and the provenance answer above |

---

## Standing requirement on any calibration filing

A filing states what it could **not** derive, not only what it found, and a cell
may not cite a run whose evidence the run itself flagged as unreliable. This is
recorded because it has bitten twice: a peer's retirement commit had its central
evidence claim refuted on review after an extraction was flagged mid-run as
shaky and used anyway, and a status cell in `docs/checks.md` was transcribed from
narrative rather than derived, standing false for an hour on 2026-08-17.

## A note on deriving claims from this repo's prose

`docs/` hard-wraps, so a single-line `grep` for a phrase longer than a few words
returns false negatives — and the verdict it inverts is "this rule is stated
nowhere", which is the one people act on by deleting things. `claim-audit` now
carries the rule and the regex-dialect trap that goes with it; read it there
rather than here, so this file does not become a second copy that drifts.

*Trimmed 2026-08-17 from a full restatement to this pointer, on the day the
skill shipped the rule. The restatement was an unwatched copy — the pattern this
file exists to catch.*

---

## Where this stands, 2026-08-17 end of session

Everything below was left green: `check_doc_links`, `check_doc_inventory`,
`check_capture_manifest`, `check_graph_discovery`, `check_node_ids`, and all five
harnesses in `bench/red/` including `spine_control`. Nothing pushed.

**Pick up here.** F1 unlocks the most — under option (b) both F3 and F4 disappear
and F5 shrinks. F7 unlocks four held items. F10 and F14 are single confirmations,
and F10's answer is "build nothing", which costs least of all.

**Not started: the postmortem-filing phase (F7, F4, F8).** Uncontested, and F7 is
its entry point: write `.postmortem.json`, decide the collection's tracked-ness for
the collection rather than per file, and move the outlier this session created in
`docs/check_postmortems.md` into the resolved directory with `filing.md`
frontmatter. `postmortem-index` resolves one directory, so a split shows partial
history as though it were complete.

**In flight with the parallel Claude session**, who owns `substrate.py`,
`provenance.py`, `bench/hwinfo.py`, `docs/hardware.md` and `CHANGELOG.md`:

- F18 step three waits on the owner's call that `bench/bench_e2e_h3.py` must
  persist at all. It persists nothing today.
- `weights.*` in the substrate block is expected to move and nothing under it
  should be required until it does. The gap worth closing is LoRA strength and
  rank, which the extractor cannot reach because strength is a node input rather
  than a filename — and reference graphs are the ones this repo most wants to
  reason about.
- They hold a blocked `_git_head` duplication between `provenance.py` and
  `substrate.py`, recorded in `docs/checks.md` rather than in memory. A ComfyUI
  restart is the scarce resource; collapse `_git_head` first because it moves no
  stamp output, and leave the version readers until the stamp adopts the substrate
  block, since reshaping those *is* a stamp schema change.

**Two things flagged and deliberately not acted on.**
`workflows/h3_probe_capture_ref3_api.json` is a committed hand-edit of a generated
file: node 23 wired to node 20, orphaning node 21 `SolAttnMiniMax`. The existing
capture is probably safe — the graph copy stored beside it is the evidence, and a
`build_workflows.py` revert cannot poison it retroactively. The risk is
prospective, and the durable fix is `sol_on=False` on the GRAPHS entry, which
fixes the UI twin too.

**Corrected 2026-08-17, after this paragraph was written: the manifest does NOT
say so.** It was cited here as evidence that the capture ran Sol-free, on the
strength of `sol_attn: bypassed_for_capture`. That value is a literal assigned
once in `bench/generate_capture_manifest.py` and never reassigned — the script
does not mention `SolAttn` anywhere, so it writes `bypassed_for_capture` whether
Sol ran or not. It is a constant wearing the shape of an observation, and it
cannot fail. The graph copy is real evidence; that field is not, and the two were
conflated. Recorded in [`docs/evidence.md`](evidence.md). That was fixed later the same day: the field is
derived now, so after a revert a regenerated manifest would report `wired` and
the record does catch it. The prospective risk is now only that the graph edit
itself reverts. And
`docs/check_postmortems.md` still holds two publication blockers, relevant only if
F7 goes tracked.

**The pattern this session kept producing, worth reading before adding anything.**
Six proposed instruments dissolved on contact with something already installed, and
four separate claims — two mine, two the peer's — passed their author's own review
and failed a second reader's. In every case the instrument had looked somewhere
other than where its author thought and returned something shaped like an answer.
`CLAUDE.md` now carries the no-new-check rule that follows from it, and the
counter-move is in `claim-audit`: name the deriving command before running it, and
check what the instrument actually examined before believing what it says.
