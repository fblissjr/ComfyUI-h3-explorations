# Sol-Attn bench plan, from 2026-08-14

What to measure next, in what order, what each run decides, and what it
predicts. Predictions are written **before** the run so a result can contradict
them; a plan that cannot be wrong is a to-do list.

Companion to `docs/open_experiments.md`, which is the opposite list — things
deliberately not measured, with the blocker for each.

## The state this starts from

Everything about the CUDA node so far is a kernel-level or arithmetic result.
**There is no end-to-end render measurement of it at all.** The e2e numbers in
`docs/SOLATTN.md` are Triton, and the ones at length 124 are below the token
floor where anything is visible.

So the first run is not a knob sweep. It is the baseline everything else is a
delta against.

## Ground rules for every run here

- **≥60,000 video tokens or the run is uninformative.** 362 frames at 1344x768
  is 107,856, near the model's ~100k ceiling. `bench_e2e_h3.py` warns below the
  floor. A null result under it reads as "this knob does nothing" when the
  truth is "this run could not have shown anything".
- **16 steps**, the shipped `SAMPLING` value, not the bench's 20 default.
- **`--runs 2` plus the discarded warmup.** Arms alternate, so drift is shared.
- **One backend per invocation.** Mixing them compares two kernels and calls it
  a knob; the bench refuses arms whose knobs the active backend lacks.
- **Restart ComfyUI after any node-code change**, then confirm the reload by
  reading the changed value back out of `/object_info` before trusting a run.
- Free the GPU (`POST /free` with `unload_models`) before any CUDA check, or it
  OOMs and looks like a regression.
- Record the kernel build. `check_sol_kernel.py` prints it; paste it in.

A render at 362 frames / 16 steps is roughly 10 minutes, so budget
`(arms x 2 + 1) x 10` minutes.

---

## Scoreboard: predictions against outcomes

Kept because a plan that is never scored is a to-do list. Run 1, 2026-08-14:

> **Every number in this table was taken at `--length 362`, which is not a
> legal length.** `h3_rules.py` applies the reference's 15.0 s ceiling after
> the frame-count snap, so 362 is 15.083 s and refused; 345 is the largest
> count on the 17n+5 grid and is what all 34 shipped API graphs carry. The
> command block below still reads 362 because that is what ran. The renders
> succeeded and nothing reported the model was out of distribution, which is
> how it happened; `bench_e2e_h3.py` now warns (`34b42b3`). Treat these as
> ratios that probably transfer — 102,816 against 107,856 video tokens, both
> far above the floor — and not as numbers for the shipped config. Run 1 is
> being redone at 345.

| question | predicted | measured (at 362, illegal) | verdict |
|---|---|---|---|
| Sol vs sage, sampler | 1.35–1.55x | **1.611x** | wrong, low |
| `centroid_tail` on vs off | 5–10% | **2.5%** | wrong, high |
| `reuse_qkv_memory` VRAM | ~1 GB | — | **uninformative**, broken instrument |

Two of three wrong, in opposite directions, which is what pre-registering is
for. The third was not a negative result: the VRAM column was reporting
torch-active bytes and could not have shown a saving.

**The scoreboard carried no length caveat until 2026-08-14**, while the
illegal length was recorded in three other files. That is the failure this
repo keeps paying for — a caveat that lives anywhere except attached to the
number it qualifies gets separated from it the first time somebody quotes the
table. A verdict column reading "wrong, low" is precisely what gets quoted.

**What the run cost that the plan did not predict:** two bench defects, both
found by running rather than reading. `bench_e2e_h3.py` had been benching
`mode="auto"` sage since 2026-08-13, and its peak-VRAM column was not
measuring device VRAM. Both are now checked (`check_bench_matches_shipped.py`)
or fixed. Budget for this: the first real run of any harness after a gap is
partly a test of the harness.

## Run 1 — the foundation, plus two knobs that ride along free

```bash
python bench/bench_e2e_h3.py --length 362 --steps 16 --runs 2 \
  --arms "sage,shipped,shipped[centroid_tail=0],shipped[reuse_qkv_memory=1]"
```

Four arms, ~90 minutes, one shared `sage` control. Three questions at once.

**Q1: is Sol worth shipping on at all, here, on the CUDA kernel?**
`sage` vs `shipped`. This is the number the project does not have.
*Prediction:* 1.35–1.55x on the sampler. The Triton path measured 1.39x at
tau 1.2 with int8 at 362 frames; the CUDA kernel routes in INT8 unconditionally
and upstream reports it 1.4x over Triton e2e, but that 1.4x was at his settings
on his box, and our tau is 1.3 rather than 1.2. If it lands under 1.2x, suspect
a silent dense fallback before believing the number — check the log for
`cuda-int8`.

**Q2: how much of the CUDA advantage is `centroid_tail`?**
`shipped` vs `shipped[centroid_tail=0]`. **This one has a deadline** — upstream
is weighing making the toggle unconditional, and if that lands the question
becomes unanswerable.
*Prediction:* 5–10% on the sampler, per upstream's own e2e figure. If it comes
out near 1.4x, then the earlier claim this repo retracted (that `centroid_tail`
*is* the CUDA-over-Triton gap) was right after all and the retraction was
wrong. Either result is worth having.

**Q3: what does `reuse_qkv_memory` buy in headroom?**
`shipped` vs `shipped[reuse_qkv_memory=1]`, read from the peak VRAM column.
Verified numerically identical, so this cannot change output — it is pure
headroom, and headroom is what gates longer clips. The heaviest shipped config
peaks at 21,186 MiB of 24,564, leaving ~3.4 GB.
*Prediction:* ~1 GB saved at this length (upstream says ~1.2 GB at 80k tokens,
and this is 108k), and sampler time within noise. If it saves nothing, the
buffer is not being reused and the flag is inert here.

**What Run 1 decides:** whether Sol stays opt-in or becomes the default; whether
`centroid_tail` needs defending upstream; and whether `reuse_qkv_memory` should
be turned on in `SOL_RECOMMENDED_CUDA`.

---

## Run 2 — `start_percent`, the knob with no justification

```bash
python bench/bench_e2e_h3.py --length 362 --steps 16 --runs 2 \
  --arms "shipped,shipped+start0.0,shipped+start0.1,shipped+start0.3"
```

Four arms, ~90 minutes. 0.4 is dropped: it costs three of sixteen steps of
sparsity, and nothing suggests the quality gain is worth that when 0.3 costs one.

`start_percent=0.2` is the only knob in the shipped config with no measured
rationale — it is the paper's number, carried through. Upstream reports a later
start affects motion least, which would make it the cheapest quality lever.

The band is **not** a step fraction. Computed for `simple` at `shift_video=12.0`:

| start | sparse steps of 16 |
|---|---|
| 0.0 | 15 (94%) |
| 0.1 | 13 (81%) |
| **0.2 shipped** | **11 (69%)** |
| 0.3 | 10 (62%) |
| 0.4 | 8 (50%) |

*Prediction:* sampler time falls roughly with the sparse-step count, so 0.0
should be ~15% faster than 0.2 and 0.3 ~5% slower. The interesting result is
quality, not time: the moving-content artifact should appear at the low end
first, and if motion really is the least-affected axis, 0.3 should buy it back
for less time than lowering tau does.

**This run cannot be judged from stills.** The failure mode is a small
persistent object dissolving partway through a clip, over about four frames. It
needs watching to the end, tracking one small object. The bench's long prompt
was written with a whip pan, brick and railings, rain texture and percussive
audio precisely so a router artifact has somewhere to show.

---

## Run 3 — `min_tokens`: CANCELLED, there is nothing to measure

Superseded 2026-08-14. This was planned as "does sparsifying the small
conditioning calls matter", on the belief that a render makes attention calls
at several sizes -- the smoke log shows 2048 and 4608 alongside 12,264.

**Those small calls are our own instrumentation.** `SageChainAssert` fires two
synthetic probes, 2048 and 4608 tokens, precisely to check the sparse gate
takes one and declines the other. They are the entire population.

H3's DiT has **exactly one** attention site, `comfy/ldm/minimax/model.py:184`,
and S there is the full packed length. Frame counts satisfy `n % 17 == 5`, so
at 1344x768 the shortest clip past 5 frames is 22 frames -> S = 7,194, already
above 4096; only a 5-frame render (S ~ 2,096) falls below. (Found by the sage
fork's claude, confirmed against the installed tree here.)

So at any real length `min_tokens` at 4096 and at 12288 select the same thing:
everything. The knob cannot change what happens, and an arm would measure
noise and report it as "within prediction" -- the worst kind of green.

Two things follow, both more useful than the cancelled run:

- **`min_tokens=4096` in `SOL_RECOMMENDED_CUDA` is harmless, not wrong.** The
  earlier note calling it "very likely wrong" was reasoning from a call
  distribution that does not exist. It only bites below 22 frames, which is
  below the token floor anyway.
- **Sage still runs 5 of 16 steps in a Sol arm.** RETRACTED 2026-08-14: this
  said sage takes "zero DiT calls" with Sol on, so a sage-config change moved
  only the sage-only arm. That was right about the `min_tokens` gate and
  ignored the **sigma window**. At `start=0.2/end=0.9`, steps 0-3 and 15 fall
  outside Sol's window; the compose gate declines them and they run sage's
  forward patch at fp16. So `mode="auto"` -> `fp16` slows BOTH arms, and the
  corrected ratio improves less than the one-sided reasoning predicted.
  Corollary worth testing: a Sol render's peak VRAM may be set by those 5
  dense steps rather than the 11 sparse ones, which would explain
  `reuse_qkv_memory` measuring nothing -- it shrinks an allocation on the
  steps that are not setting the peak.

## Run 4 — `sink_conditioning` at reference load

**Blocked on a build.** `bench_e2e_h3.py` is t2v-only — no reference wiring at
all. This needs `LoadImage` → `MiniMaxH3ReferenceFit` → the r2v path, as a
`--refs` axis, so a paired same-seed A/B is still possible.

It is the highest-value unmeasured question and the most expensive to set up.
Arithmetic over the measured row counts in `docs/h3_references.md` says one
345-frame video reference at `exact_kv_and_rows` forces **57.9%** of attention
exact, against 35.1% at `exact_kv` — a 23-point swing, where `start_percent`
0.2 → 0.3 is one step of sixteen.

*Prediction:* at one image reference at `match` the two settings are within
noise; with a video reference `exact_kv` is worth 15–25% of sampler time, and
the cost shows up in generated audio, which is the reason
`exact_kv_and_rows` is on.

---

## Run 5 — re-baseline the frontier above the floor

Everything in `docs/SOLATTN.md`'s frontier table is at length 124 = 37,296
tokens. Until this runs, most of that page cannot be quoted for the CUDA path.

Lowest priority not because it does not matter, but because Runs 1–3 produce
most of it as a side effect.

---

## Deliberately not planned

- **CUDA vs Triton e2e, ours.** Confirmatory only: the migration already
  happened, on an accuracy argument that does not depend on the ratio. One arm
  of `shipped_triton` under `--sol-backend triton` gets it whenever someone
  wants it.
- **`routed_cap_percent`.** Upstream reports ~30 as lossless with 3x headroom.
  It trades quality for memory, and `reuse_qkv_memory` addresses memory without
  touching quality, so measure that first and come back only if headroom is
  still the binding constraint.
- **`tau` re-sweep.** Already measured at 362 frames on Triton, and the artifact
  onset near 1.5 is the binding constraint rather than the timing.
