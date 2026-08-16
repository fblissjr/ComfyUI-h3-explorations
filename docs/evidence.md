# What holds, and what does not

Last updated: 2026-08-16.

`docs/checks.md` indexes what is *checked*. `docs/open_experiments.md` indexes
what is *not measured*. This file is the third case and the one that kept
biting: claims that **were** measured, are written down as numbers, and should
not be relied on — because the conditions they were taken under are not the
conditions we ship.

An entry leaves this file when someone re-measures it, not when someone
re-reads it.

## Relationship to `SOLATTN.md`'s header

Both were written on 2026-08-14, in parallel, by two sessions that did not
know the other was doing it — which is its own entry in the postmortem.

The split, so neither drifts: **SOLATTN.md's header is the Sol-Attn quick
reference**, read by someone about to quote a number off that page. **This
file is the repo-wide ledger** and owns the three things a topic page cannot:
claims that are not about Sol-Attn (the fp8/fp16 accuracy figure is a sage
claim; the smoke's prompt substitution is a harness one), the **what restores
it** column, and the environment section below — the venv and `coderef/` are
not in git and nothing else records what changed there.

If the two disagree, this file is wrong until proven otherwise: SOLATTN.md is
maintained by whoever is running the measurements.

## Why this file exists

On 2026-08-14 two sessions working the same repo found nine separate claims in
this class, and **not one of them was caught by whoever wrote it**. Every one
came from a second reader. The pattern was identical each time: a number
measured under one configuration, carried into another, with the qualifying
sentence living in a different file — or in the same file, one paragraph away
from the table.

That is caveat decay, and `docs/checks.md` already names it. What it did not
have was somewhere for the *surviving* list to live, so each caveat was
attached to whichever page happened to discuss it, and none of them were
attached to the number.

---

## Do not rely on these

| claim | why not | what restores it |
|---|---|---|
| **Sol vs sage = 1.611x** | Wrong on **two** axes, not three: an **fp8** sage baseline the graphs do not ship, and sage running 5 of 16 steps inside a Sol arm so the fp16 fix moves **both** arms. **The length objection is WITHDRAWN 2026-08-14** — 362 is trained; only the reference pipeline declines it | rerun with the shipped fp16 baseline; length was never the problem |
| **`centroid_tail` = 2.5% e2e** | Ours, two runs, 0.1% spread, at 362 — **which is fine.** Retained here only because it shares Run 1's fp8 baseline | rerun on the fp16 baseline |
| **Upstream's `centroid_tail` ~5–10% e2e** | Upstream conversation, his box, his settings. Never reproduced here, and our own 2.5% disagrees | reproduce, or drop the figure |
| **CUDA is 1.4x over Triton e2e** | Upstream conversation. Never reproduced here | a paired run on this box |
| **`centroid_tail` is "~1.4x", or "the biggest speed knob in the node"** | **Tripwire row, and the trap has now caught two readers.** The node's own tooltip says "~1.4x faster" and that figure is the **operation**, not end-to-end. Ours measured **2.5% e2e** (row above), upstream reports 5–10%. Against the shipped frontier — sol 1.20x, +int8 1.39x, tau 1.6 1.53x — `centroid_tail` is the **smallest** knob in the node, not the biggest. `docs/SOLATTN.md` recorded itself making this exact conflation on 2026-08-14; an analysis on 2026-08-16 made it again and built an argument on top of it (that a token ordering earns its keep by making the "1.4x knob" safe to leave on — the mechanism is real, the prize is 2.5%). **The tell is the unit**: a knob's op-level speedup and a config's e2e speedup share a denominator with nothing. **Deliberately NOT added to the `retraction-consumers` block** — "1.4x" is correct where it appears (the node tooltip, upstream's CUDA-vs-Triton figure, `SOLATTN.md`'s knob table), so hunting the string would fire on every correct use and train readers to skim the output. Same shape as the "2.7x" note below: one spelling, two claims, and only the pairing is wrong. | nothing to restore — the correct figures already exist. This row is here so the next person who reads the tooltip finds the correction first |
| **`reuse_qkv_memory` saves nothing** | Not a negative result. The VRAM column was reporting torch-active bytes and could not have seen it | fixed instrument (`f1dff99`), rerun |
| **any peak-VRAM figure between 13:08:49 and `f1dff99`** | An external write reverted the device poller on disk; a `git add -A` then committed the reverted state | re-measure |
| **any fp8-vs-fp16 sage accuracy ratio**, spelled here once as **"2.7x more accurate"** so the tripwire below has something to match | **WITHDRAWN 2026-08-16 by the owner, and deleted rather than caveated** — the only entry in this table handled that way. This row is now the sole place the phrase appears in the repo. Two competing ratios had accumulated (a synthetic `torch.randn` sweep run here, and a smaller figure reported secondhand from the sage fork's captured activations) plus the `mean_rtol` values behind both. Ruled out on provenance, not on size: the sweep measures an input distribution H3 does not have; the real-activation figure was never re-derived here and its script is not committed in the fork, so it was an uncommitted ad-hoc run cited across a repo boundary; and nothing in `bench/` uses captured activations, so a rerun today would reproduce the synthetic instrument rather than replace it. The **decision** to ship `fp16 (most accurate)` is unaffected — it rests on the owner's perceptual verdict, which never depended on a ratio. **`attention.py` and `README.md` keep a "2.7x" that is a *speed* figure against torch's flash backend — a different, correct claim. Two numbers, one spelling; do not sweep it.** | grade a sage kernel against `~/Storage/h3_captures/2026-08-15_dense_124f_1344x768/`, which exists and has never been used for this. Until then this repo has no accuracy figure and must not acquire one by inference |
| **"the LoRA arm carried the reference's studio lighting onto his cheek"** | **Retracted 2026-08-16, same day, by the author.** The OBSERVATION is sound and triple-confirmed (owner blind, me from stills, Gemini blind): one arm lights the man with a hard key shadowing his right cheek, the other softly and evenly. The EXPLANATION was invented. `ref_1` is soft near-frontal studio light with no defined shadow edge anywhere on the face — **there was no shadow to carry**. The premise was never checked before the mechanism was built on it, which is the failure this file exists for, committed hours after writing a doc about it. Gemini's reading is the better one: a hard key is defensible as low golden-hour sun through the window, so the lighting difference is two valid interpretations of the brief rather than one arm failing it | nothing restores it. What survives from that render is parallax coherence and identity equivalence, both recorded in `docs/roadmap.md` |
| **any reference measurement taken before `_ref_prompt()` is fixed** | **Common-mode confound, named 2026-08-16.** Not one number — a floor under all of them. The generator asserts Subject 2 carries "architecture, palette, and lighting" regardless of the reference, and that assertion is BINDING: both arms of the LoRA A/B built a house that appears in no reference image. The same template specifies an interior soundscape over an outdoor scene and gives two contradictory lighting instructions. Separately all 20 reference prompts are single-shot at 46–73 words of `detailed_description` against the guide's 350–500, so the model improvises most of the clip. None of this invalidates a PAIRED comparison — the prompt is identical across arms — but it bounds what any reference result generalises to, and it means "the model did X" may be "the prompt asked for X" | fix `_ref_prompt()` (roadmap step 0), then re-measure |
| **Sol is 0.999919 accurate** | Implementation fidelity, not total error — the harness compares kernel against reference **at the same tau**, so the sparse approximation is on both sides and cancels. Also `T=512`, and the O(T²) reference cannot run at real length | the Sol-vs-dense diagnostic (`44becf0`), once the card frees |
| **text = 38 rows, sequence = 12,264** | `smoke_h3.py` substitutes **both** the prompt (27 words, against the graph's 216) and the length. Not a scaled-down shipped graph — text does not scale with length and was replaced | one preflight on a shipped graph, unmodified |
| **everything derived from that**: audio dominates the sink; text is the whole v2 narrowing; `sink_q` start is 0 on t2v | all smoke-harness statements. On the shipped graph, text extrapolates to ~304 rows / 4 blocks — still an extrapolation from one point | same preflight |
| **the sink's audio framing** | A **t2v** framing. In reference-heavy graphs the sink is overwhelmingly *reference* rows, and a video reference's failure mode is motion drift, not the thinness argument the knob is named for | measurement at reference load; the bench has no `--refs` axis yet |
| **reference-load table: 35.1% / 57.9%** | Wrong on three axes — v1 formula (v2 stops running reference queries dense), 362 frames, 1344x768. The shipped reference arms were 345 at 1024x768, and are 362 since 2026-08-16 | redo as v1-vs-v2, measured not derived |
| **"with Sol on, sage gets nothing"** | Retracted 2026-08-14. Reasoned from `min_tokens` and forgot the sigma window. Sage runs 5 of 16 steps | — corrected in place |
| **`min_tokens` 4096 is "very likely wrong"** | Retracted, and the retraction's reasoning was itself corrected 2026-08-14: it is one `optimized_attention` call in source but **52 modules** through it. Conclusion survives — the 50 DiT blocks run at the full packed length, above both thresholds, and the 2 refiner blocks run at the text span, below both — so 4096 and 12288 still select the same thing | — corrected in place |
| **"fp16 lands on the steps where precision matters most"** | Inference, not measurement. `start_percent` has never been measured at any length on either backend | measure `start_percent` (Run 2, not started) |
| **any bench progress read from its own stdout mid-run** | The warmup `print` lacks `flush=True`, and `tee` makes stdout block-buffered, so finished lines sit in the buffer. Read ComfyUI's progress lines instead | add `flush=True` |
| **Morton is "worth 1.16x alone", at "94% GPU utilisation"** | Retracted 2026-08-16. Triton, 362 frames, stacked on int8 — correct for what it measured, and not a description of the CUDA backend, where the permutation is free. It had been the standing argument for `morton=False` in `h3_config.py` and in `docs/SOLATTN.md`'s Configuration findings, and the two disagreed for several hours | a CUDA Morton arm at fixed tau, if anyone wants a cost number at all |
| **the CUDA replacement, "0.8 s of 861, or 1.0009x"** | **Do not quote this either**, and it is the more instructive of the two. It is one arm of a two-arm control presented as the isolated number: the dense pair moved +0.8 s and the sparse pair moved −1.2 s, i.e. opposite signs, both at or under the bench's measured run-to-run spread on one run per arm. Morton-on cannot be faster, so the pair is measuring noise. The correct claim is unquantified — **free** | more runs per arm, if the cost is ever worth pinning. Nothing currently depends on it |

**362 is the max length, and every "345 is legal / 362 is illegal" claim is
withdrawn. Owner decision, 2026-08-16.** 345 is the largest count the
*reference pipeline* will emit — its `max_duration` is a hard-coded 15.0 s and
362 is 15.083 s — which is a fact about diffusers. This repo turned that into
"362 is illegal / out of distribution" for most of 2026-08-14 and withdrew a
bench run over it; that was an inference from a validator, not a fact about the
checkpoint, and the framing survived the first correction because 345 stayed
the default.

`LONG_LENGTH` is now 362 and all shipped graphs carry it. `duration_in_range()`
is the model's window; `reference_would_emit()` is the separate portability
question.

**What 362 rests on, stated because this table exists for exactly this.** One
upstream statement recorded 2026-08-14 (`6e85e48`) with no artifact attached,
plus LightX2V shipping a 362-frame config. MiniMax's README gives a rounded
"4-15 seconds" and the official checkpoint configs state no frame limit at all,
so the primary source neither confirms nor refutes it. This is a decision taken
on thin evidence, not a measurement — do not quote it as one.

---

## These hold

Kept short on purpose — every row is something a second reader confirmed or an
instrument that has been shown red.

- **The CUDA seam works.** Live render, `cuda-int8` in the log, override
  chained, 50 forwards composed. Not a source read.
- **One `optimized_attention` call in source**, `comfy/ldm/minimax/model.py`,
  reached by **52 modules** — 50 DiT blocks plus 2 token-refiner blocks, all
  sharing one `Attention.forward` (`num_layers=50`,
  `token_refiner_num_layers=2`). "One call site" is not "one call", and the
  refiner blocks see only the text span while the 50 see the full sequence.
- **Sage runs 5 of 16 steps under Sol** at the shipped window — verified at two
  layers of the node source and cross-checked against this repo's own
  sigma-window table, which independently gives 11/16 sparse.
- **`reuse_qkv_memory` is numerically identical** to the normal entry, six
  digits. It cannot change output. What it *buys* is unmeasured.
- **Both backends are arithmetically equivalent** at `T=512` fidelity.
- **All 34 API graphs carry `LONG_LENGTH`**, read from their widgets.
- **Every `bench/check_*.py` passes**, and the ones added today were each shown
  red first: `check_sol_kernel` (4 cases), `check_bench_matches_shipped`,
  `vendored`, `node_version`.
- **The node is vendored, symlinked and hash-pinned**, so the file ComfyUI
  loads cannot drift from the tracked one, and an unrecorded hash fails rather
  than warns.

---

## Enumerated consumers, checked mechanically

A retraction is done when every **consumer** of the claim is enumerated, not
when the claim is corrected. That is today's most-repeated failure in one
sentence: "sage gets nothing" was retracted and went on producing wrong answers
in three more places, and the 23-point swing was retracted in this file while
still carrying an argument in `docs/bench_plan.md`.

The block below makes that enumeration executable.
`bench/check_retraction_consumers.py` fails when a retracted phrase appears in a
file **not** listed for it. There is no caveat-detection and no judgement: a
mention inside a `RETRACTED:` block and a live use are indistinguishable to a
matcher, so the check does not try. It answers one decidable question — has this
claim reached a file nobody signed off on — which is the failure mode that
actually occurred, four times, all in files that acquired the claim *after* the
retraction.

Adding a file here is a claim that someone read that occurrence and it is
correct in context. Do not add one to silence the check.

```retraction-consumers
ONE LINE PER FIELD. The parser reads only the line beginning `ALLOW:`, so a
wrapped continuation is silently dropped and the row enumerates fewer files
than it appears to. That fails closed -- the missing files show up as
unlisted consumers -- but it wastes a run, which is how it was found on
2026-08-16.

PHRASE: 2.7x more accurate
ALLOW: docs/evidence.md
WHY: the figure was DELETED repo-wide on 2026-08-16, not caveated, so the
     allowlist collapses to this ledger. The phrase is retained here so the
     check keeps a tripwire on it: if it reappears in any file, someone has
     reintroduced a withdrawn number. Still deliberately NOT matching bare
     "2.7x" -- attention.py and README.md use that string for a speed figure
     against torch flash, a different and correct claim.

PHRASE: 23-point swing
ALLOW: docs/evidence.md docs/bench_plan.md docs/SOLATTN.md
WHY: bench_plan's is inside the retraction block that replaced the argument;
     SOLATTN's are its "Do not rely on" table and the reference section.

PHRASE: sage gets nothing
ALLOW: docs/evidence.md docs/SOLATTN.md
WHY: both are retraction statements naming it as withdrawn.

PHRASE: zero DiT calls
ALLOW: docs/bench_plan.md docs/checks.md
WHY: bench_plan's is inside a "RETRACTED 2026-08-14" bullet stating the
     5-of-16 correction. checks.md quotes it as the worked example of why this
     check is an allowlist rather than caveat-detection -- and it was caught
     by the check on the commit that added it, which is the behaviour wanted.

PHRASE: exactly one attention site
ALLOW: docs/SOLATTN.md
WHY: the sentence that contains it says the page claimed this until 2026-08-14
     and corrects it to 52 modules.

PHRASE: 38 text rows
ALLOW: docs/SOLATTN.md
WHY: both name it as a smoke-harness figure rather than a shipped-graph one.

PHRASE: worth 1.16x alone
ALLOW: docs/evidence.md docs/morton.md docs/SOLATTN.md docs/open_experiments.md docs/bench_plan.md CHANGELOG.md
WHY: Triton, 362 frames, stacked on int8, retracted 2026-08-16 when a dense
     control showed the CUDA permutation free. morton.md owns the
     retraction; SOLATTN.md's is the "do not rely on" row and the corrected
     Configuration-findings bullet; open_experiments.md and bench_plan.md
     name it as withdrawn in the paragraphs that used to argue from it;
     CHANGELOG.md records the retraction as history and is not rewritten.
     Enumerated the same day the retraction was made -- the earlier three
     same-session retractions were not, and one of them (this one) went on
     being asserted in SOLATTN.md's Configuration findings for hours
     afterwards. That is the open question raised in
     internal/postmortems/2026-08-15_session_morton-sampler-and-refs.md,
     answered by the failure recurring rather than by argument.

PHRASE: 94% GPU utilisation
ALLOW: docs/evidence.md docs/morton.md docs/SOLATTN.md docs/open_experiments.md CHANGELOG.md
WHY: the same measurement and the same retraction. Listed separately because
     the two halves of that claim were quoted independently -- h3_config.py
     carried the utilisation figure as the mechanism story for the 1.16x, so
     either half can reappear without the other.
```

**What this does not defend, and the gap is not small.** The sharpest decay
found on 2026-08-14 has no string to match: `docs/bench_plan.md` read "one
**345-frame** video reference", where 345 is the *reference* length and looks
like the shipped config, concealing that the *target* was 362. The retracted
thing was the **pairing** of two numbers, not any token in the sentence. No
phrase matcher can see that, and a check that appears to cover a class it
silently omits is how the 2026-08-13 validator bug worked — so it is said here
rather than left to be assumed.

**The general form, named 2026-08-16 after it happened again.** This check
defends the *spelling* of a retracted claim, not the claim. A retracted
**measurement** can reappear in any wording containing no listed phrase, and
**derived figures are exactly that shape** — a spread between two withdrawn
numbers, a ratio of them, a rescale, a rounding. The instance: after the
accuracy ratios were withdrawn, `CLAUDE.md` kept a sentence saying all three
fp8 variants land within a stated tolerance of each other — a mean_rtol spread
from the same withdrawn sweep — nine lines below the paragraph announcing there
was no numeric half left. The commit that removed its neighbours left it, and
its author read that paragraph twice the same day. Two properties made it
invisible: it carried no retracted token, and it read as a conclusion rather
than as a number.

So when withdrawing a measurement, grep for the *quantities* it produced and
not only for the sentence it was written in.

**And it cannot be closed by adding a row, which was tried on 2026-08-16 and
reverted the same hour.** The ledger is an allowlist: every phrase needs at
least one file that legitimately contains it, and the `stale_allowlist` case
warns when a listed phrase is absent from its own allowed files. A spelling
that should appear **nowhere** therefore cannot be expressed — listing it emits
a permanent warning on every run, which is the crying-wolf failure this repo
treats as worse than no check. The model fits a claim with a legitimate home
and a forbidden everywhere-else; it does not fit a claim that is simply banned.
Catching those needs a different instrument, and until one exists this
paragraph is the instrument.

---

## Environment, because it is not in git

The venv and ComfyUI were both changed today and neither is version-controlled
here:

- `comfy-kitchen` replaced: `0.2.31` → `0.2.31+sol.c04ef20`, **built from
  source for sm_89 only**. It will not work on another architecture, and it
  declares a version the stock wheel also declares.
- `nanobind` 2.14.0 added.
- `custom_nodes/ComfyUI-SolAttn-cuda/` created; its `sol_attn_minimax.py` is a
  **symlink into this repo's `vendor/`**. Editing through the installed path
  writes into the tracked file.
- `coderef/comfy-kitchen-sol/` cloned at `c04ef20` with submodules, and its
  `pyproject.toml` **edited** — that is a modification to a checkout of
  someone else's tree.

Record the node hash and the `comfy-kitchen` tag with any measurement. The
branch rebases and the build declares a stock version number, so nothing else
can tell two builds apart.
