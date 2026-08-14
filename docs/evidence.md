# What holds, and what does not

Last updated: 2026-08-14

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
| **`reuse_qkv_memory` saves nothing** | Not a negative result. The VRAM column was reporting torch-active bytes and could not have seen it | fixed instrument (`f1dff99`), rerun |
| **any peak-VRAM figure between 13:08:49 and `f1dff99`** | An external write reverted the device poller on disk; a `git add -A` then committed the reverted state | re-measure |
| **fp8 is 2.7x less accurate than fp16** | **Synthetic input.** The sage fork measures 1.3x on real captured H3 activations and calls every synthetic rtol a pessimistic bound. `bench/bench_minimax_attn.py` builds `torch.randn`; nothing in `bench/` uses captured activations, and our 0.0969–0.0984 matches the fork's synthetic 0.098, not its real-activation 0.026. **Do not grep-and-replace "2.7x"** — `attention.py` and `README.md` use the same string for a *speed* figure against torch's flash backend, which is a different, correct claim. Two numbers, one spelling | the capture hook is written (`756a65e`) and has never been run. Note it currently captures with Sol on, which yields steps 0-3 and 15 only — both ends of the schedule, none of the middle |
| **Sol is 0.999919 accurate** | Implementation fidelity, not total error — the harness compares kernel against reference **at the same tau**, so the sparse approximation is on both sides and cancels. Also `T=512`, and the O(T²) reference cannot run at real length | the Sol-vs-dense diagnostic (`44becf0`), once the card frees |
| **text = 38 rows, sequence = 12,264** | `smoke_h3.py` substitutes **both** the prompt (27 words, against the graph's 216) and the length. Not a scaled-down shipped graph — text does not scale with length and was replaced | one preflight on a shipped graph, unmodified |
| **everything derived from that**: audio dominates the sink; text is the whole v2 narrowing; `sink_q` start is 0 on t2v | all smoke-harness statements. On the shipped graph, text extrapolates to ~304 rows / 4 blocks — still an extrapolation from one point | same preflight |
| **the sink's audio framing** | A **t2v** framing. In reference-heavy graphs the sink is overwhelmingly *reference* rows, and a video reference's failure mode is motion drift, not the thinness argument the knob is named for | measurement at reference load; the bench has no `--refs` axis yet |
| **reference-load table: 35.1% / 57.9%** | Wrong on three axes — v1 formula (v2 stops running reference queries dense), 362 frames, 1344x768. The shipped reference arms are 345 at 1024x768 | redo as v1-vs-v2, measured not derived |
| **"with Sol on, sage gets nothing"** | Retracted 2026-08-14. Reasoned from `min_tokens` and forgot the sigma window. Sage runs 5 of 16 steps | — corrected in place |
| **`min_tokens` 4096 is "very likely wrong"** | Retracted, and the retraction's reasoning was itself corrected 2026-08-14: it is one `optimized_attention` call in source but **52 modules** through it. Conclusion survives — the 50 DiT blocks run at the full packed length, above both thresholds, and the 2 refiner blocks run at the text span, below both — so 4096 and 12288 still select the same thing | — corrected in place |
| **"fp16 lands on the steps where precision matters most"** | Inference, not measurement. `start_percent` has never been measured at any length on either backend | measure `start_percent` (Run 2, not started) |
| **any bench progress read from its own stdout mid-run** | The warmup `print` lacks `flush=True`, and `tee` makes stdout block-buffered, so finished lines sit in the buffer. Read ComfyUI's progress lines instead | add `flush=True` |

**362 is a trained length. Corrected 2026-08-14.** The reference *pipeline*
refuses it — its `max_duration` is a hard-coded 15.0 s and 362 is 15.083 s —
but that is a spec ceiling landing one grid step short of the real training
maximum, not a statement about the model. 362 is the longest length H3 was
trained on. This repo called it "illegal" and "out of distribution" for most
of 2026-08-14, which was an inference from a validator, not a fact about the
checkpoint.

All 34 shipped API graphs are at 345 (`LONG_LENGTH`), which stays a reasonable
default — it is inside the reference's window, so a graph exported from here
runs in diffusers too. That is a portability argument, not a quality one.

---

## These hold

Kept short on purpose — every row is something a second reader confirmed or an
instrument that has been shown red.

- **The CUDA seam works.** Live render, `cuda-int8` in the log, override
  chained, 50 forwards composed. Not a source read.
- **345 legal / 362 illegal**, from `h3_rules.py` and the reference.
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
- **All 34 API graphs carry 345**, read from their widgets.
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
PHRASE: 2.7x more accurate
ALLOW: workflows/h3_config.py CHANGELOG.md CLAUDE.md
WHY: all three carry the synthetic-input caveat inline. Deliberately NOT
     matching bare "2.7x" -- attention.py and README.md use that string for a
     speed figure against torch flash, a different and correct claim.

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
```

**What this does not defend, and the gap is not small.** The sharpest decay
found on 2026-08-14 has no string to match: `docs/bench_plan.md` read "one
**345-frame** video reference", where 345 is the *reference* length and looks
like the shipped config, concealing that the *target* was 362. The retracted
thing was the **pairing** of two numbers, not any token in the sentence. No
phrase matcher can see that, and a check that appears to cover a class it
silently omits is how the 2026-08-13 validator bug worked — so it is said here
rather than left to be assumed.

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
