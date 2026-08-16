# Sol-Attn on MiniMax H3

Sol-Attn ([arXiv 2607.24027](https://arxiv.org/abs/2607.24027)) is training-free
block-sparse attention. Each 64-token query block attends a routed subset of key
blocks exactly and covers the rest with one pooled term per block, so the whole
sequence still contributes to the softmax denominator.

**Two implementations exist and this page covers both.** As of 2026-08-14 the
CUDA one is what every shipped graph wires and what this repo measures against.
The Triton one is what every number older than that date was taken on. It was
**moved out of `custom_nodes/` into `coderef/` on 2026-08-16**, so it no longer
registers nodes and cannot be wired by accident — but it is kept, not retired,
and two live tooling dependencies still need the directory. See "The Triton
node". They are not interchangeable and they do not share a knob vocabulary —
see `SOL_RECOMMENDED_CUDA` in `h3_config.py` for what the migration did and did
not carry over.

Everything here is single-machine (RTX 4090, sm_89), single-workload. The sage
baseline is [SageAttention-ada](https://github.com/fblissjr/SageAttention-ada),
a fork, never measured against stock SageAttention — so every ratio means "on
this fork, this model, this box".

---

## Read this before quoting any number on this page

Status 2026-08-14, end of session. This page changed more in one day than in
the two weeks before it, and a large share of what it used to say is now
labelled rather than deleted. Read this list first.

### Do not rely on

| claim | why not |
|---|---|
| every e2e timing from Run 1, incl. **1.611x** | taken against an **fp8** sage baseline the graphs do not ship, and before it was known that sage runs 5/16 steps in the Sol arm. Wrong on three axes. |
| `centroid_tail` = **2.5%** | the measurement is sound (two runs, 0.1% spread) but it is at 362. Ordering is trustworthy; the magnitude is not pinned to a shipped config. |
| `reuse_qkv_memory` "no VRAM saving" | **uninformative, not negative.** The peak-VRAM column was reporting torch-active bytes, which resolved only the resident-weight plateau. The instrument could not have shown a saving. |
| any fp8-vs-fp16 sage accuracy ratio | **withdrawn entirely 2026-08-16**, not caveated. The figures are removed from this repo; `bench_minimax_attn.py` builds `randn` and nothing here uses captured activations, so no accuracy number measured today would be defensible either. See `docs/evidence.md`. |
| **0.999919** and the other correctness cosines | **implementation fidelity, not accuracy.** Kernel is graded against the reference *at the same tau*, so the sparse approximation is on both sides and cancels. Not comparable to any dense kernel's accuracy number. Also T=512 synthetic. |
| **1.4x** CUDA over Triton e2e | upstream conversation. Never reproduced here. |
| `centroid_tail` **5-10%** e2e | upstream conversation. Ours measured 2.5%. |
| "38 text rows", "sequence 12,264" | from `smoke_h3.py`, which **substitutes both the prompt and the length**. Not a scaled-down version of any shipped graph. Everything derived from it -- that audio dominates the sink, that text is the whole v2 narrowing, that `sink_q` start is 0 -- is a statement about the harness. |
| the reference-load table's **35.1% / 57.9%** | wrong on three axes for what we ship: a **v1** formula (v2 changed the mechanism), at **362**, at **1344x768** where the ref graphs are 1024x768. |
| "with Sol on, sage gets nothing" | **retracted.** Sage runs **5 of 16 steps** -- the sigma window, not just `min_tokens`. |
| bench progress read from its own stdout mid-run | the warmup `print` lacks `flush=True`; under `tee` stdout is block-buffered. A finished render can read as "still in warmup" for 20 minutes. Read ComfyUI's log instead. |
| **any quality A/B on this page, as a like-for-like** | all of it was taken on `res_multistep`. The default sampler is `er_sde` since 2026-08-15, which injects noise every step. A knob that perturbs attention numerics reads as more "reseeded" under it. Timings should carry (both are one eval per step); quality comparisons are owed a re-run. |

### Can rely on

All verified on this box today, by running rather than reading:

- **The CUDA seam works.** Live render, `cuda-int8` in the log, sage's override found and chained, 50 forwards composed.
- **345 is legal, 362 is not.** `h3_rules.py` and diffusers `before_denoise.py`, independently.
- **H3's DiT reaches `optimized_attention` from one source line but 52 modules.**
  `comfy/ldm/minimax/model.py`: the 50 `DiTBlock.attn` at the full packed length,
  plus 2 `TokenRefiner` blocks on the text tokens alone (311 rows in the shipped
  graph). This page said "exactly one attention site" until 2026-08-14 — true of
  the line, misleading about the modules, and the refiner pair is exactly what
  keeps `ModelAttentionBackend` from being a no-op. The VAE has its own call
  (`comfy/ldm/minimax/vae.py`) that no override can reach: it passes no
  `transformer_options`, so `wrap_attn` skips the lookup.
- **Sage runs 5 of 16 steps with Sol on** at the shipped window -- verified at both the compose gate and the override, and cross-checked against this page's own 20-step figure of 6 dense steps.
- **The two backends are arithmetically equivalent** at T=512 fidelity, each graded in its own measured `centroid_tail` mode.
- **`reuse_qkv_memory` cannot change output** -- numerically identical to the normal entry, six digits.
- **v1 -> v2 is schema-identical**, so no graph regeneration is needed across that upgrade.
- Every check added today was **shown red** before being trusted.

### Measured 2026-08-14, late: the first shipped-graph segment breakdown

Every segment figure this repo had cited came from `bench/smoke_h3.py`, which
substitutes **both** the prompt and the length. This is `h3_probe_sol_on_api.json`
submitted **as shipped** — 345 frames, 1344x768, the graph's own 216-word prompt
— with only `verbose` changed:

```
sequence length 104,277
  video   102,816   98.6%
  audio     1,150    1.1%
  text        311    0.3%
```

So `video_start = 1,461` and the conditioning region is **23 blocks** of 64.

**This verifies v2's `sink_q` on real geometry**, which is the gap upstream
flagged as exercised only synthetically. Predicted from the layout:
KV `(0, ceil(1461/64)) = (0, 23)`, and v2's query span starting at
`text_len // 64 = 311 // 64 = 4`. Logged:

```
conditioning sink: KV blocks (0, 23) exact, dense query blocks (4, 23)
```

Exact match. KV untouched, query span narrowed by 4 blocks of 23 — v1 would
have read `(0, 23)`. A start of 4 rather than 0 also distinguishes "v2 engaged"
from v2's own silent `audio is None` fallback, which returns the v1 range.

Two things follow. **On t2v the saving is real but small**: 4 blocks of 23 in a
region that is itself 1.4% of the sequence. And **the earlier claim that a
short-prompt t2v graph cannot discriminate v2 was wrong** — it assumed 38 text
rows, which was the smoke harness's substituted prompt. The shipped graph has
311.

The full packed sequence goes through the CUDA kernel at the shipped length:
`sparse (1, 104277, 56, 128) tau=1.3 cuda-int8`.

### Never measured, still

`start_percent` and `end_percent`, at any length, on either backend. The
segment breakdown of any graph **as shipped**. Sol's accuracy against dense
attention at production sequence length, on real activations.

## Start here

### 1. Pick a backend

| | CUDA | Triton |
|---|---|---|
| node id | `SolAttnMiniMax` | `SolAttnPatch` |
| pack | `custom_nodes/ComfyUI-SolAttn-cuda/` | `coderef/ComfyUI-SolAttn_triton/` (moved out of `custom_nodes/` 2026-08-16; not registered) |
| needs | a source build of `comfy_kitchen`'s `sol_attn` branch | nothing beyond Triton |
| speed | upstream reports **1.4x over Triton at the same tau, end to end** | baseline |
| accuracy vs the algorithm's reference | 0.999919 | 0.999885 (int8), 0.999995 (bf16) |
| default tail mode | `centroid_tail=True` | per-row, not adjustable |
| status here | **what every shipped graph wires, and what new work measures** | kept for reproducing pre-2026-08-14 numbers |

**Use CUDA.** The backends are arithmetically equivalent (see below), so there
is no accuracy argument for Triton, and CUDA is faster.

**But do not uninstall Triton, and the reason is not sentimental.** It carries
two live dependencies and one historical one; only the third is a benchmark
concern. See "The Triton node" below for the full statement — the short version
is that `bench/check_solattn_correctness.py` **hard-requires** it and
`SolAttnBlockProbe` has no CUDA equivalent.

### 2. Install the CUDA kernel

`comfy_kitchen.sol_attn` **ships on no wheel.** It exists only on
kijai/comfy-kitchen's unmerged `sol_attn` branch. The stock
`comfy-kitchen==0.2.31` that ComfyUI pins has no `sol_attn` at all.

```bash
git clone -b sol_attn https://github.com/kijai/comfy-kitchen.git
cd comfy-kitchen
git submodule update --init --depth 1 third_party/flash-attention third_party/cutlass
uv pip install 'nanobind>=2.0.0'
COMFY_CUDA_ARCHS=89 uv build --wheel --no-build-isolation .   # your arch here
uv pip install --force-reinstall --no-deps dist/comfy_kitchen-*.whl
```

Three things that will bite:

- **`--no-build-isolation` is required**, or arch detection sees no GPU and
  compiles every target. It also means build deps are not installed for you —
  `nanobind` is the one that is missing.
- **Submodules are not optional.** `flash_decode.cu` includes `flash.h` from
  `third_party/flash-attention`, and a fresh clone fails there.
- **The branch declares version `0.2.31`**, identical to the wheel ComfyUI
  pins. Tag your build (this box uses `0.2.31+sol.c04ef20`) or nothing can
  tell the two apart.

Requires bf16, head_dim 128, sm_80+.

### 3. Verify it, because failure is silent

If the kernel is swapped out after the fact — a `--force-reinstall`, a Manager
repair, a fresh venv — **nothing errors**. The override declines every call and
falls back to dense. The render succeeds and is merely slower and numerically
different.

```bash
python bench/check_sol_kernel.py --require     # kernel present, right backend, signature intact
python bench/check_solattn_correctness.py      # both kernels against the eager reference
python bench/smoke_h3.py --workflow h3_probe_sol_on_api.json --log <comfyui.log>
```

The smoke is the only one that submits a prompt. Its three log lines are the
composition check: sage engaged, Sol found sage's override already installed
(the ordering check), and sparse actually ran.

**The CUDA seam is verified as of 2026-08-14**, and it is a different code
path from Triton's — `_apply_patch`, `_compose_module_patch` and
`_install_compose_hooks` are the CUDA node's own. What a passing run looks
like, from the log:

```
[sol_attn] chaining onto an existing attention override
[sol_attn] composed with 50 patched attention forward(s)
[sol_attn] dense (1, 2048, 56, 128): seq 2048 < 4096
[sol_attn] sparse (1, 4608, 56, 128) tau=1.3 cuda-int8
[h3] chain assert: sage routed a 2048-token probe on fp16_cuda and correctly
     did NOT get the 4608-token one, so the sparse gate at 4096 is live
[sol_attn] conditioning sink: KV blocks (0, 3) exact, dense query blocks (0, 3)
```

**Check the kernel tag.** `cuda-int8` is the CUDA kernel; Triton logs
`int8 pointer`. That string is the difference between the kernel running and a
silent fallback.

### 4. Know where the gains live before you measure

Video tokens are `latent_t * (w/32) * (h/32)`, with
`latent_t = ((n - 5) // 17) * 5 + 2`:

| canvas | 73 frames | 124 | 250 | 300 | **345** | 362 |
|---|---|---|---|---|---|---|
| 1344x768 (1008/frame) | 22,176 | 37,296 | 72,576 | 87,696 | **102,816** | 107,856 |
| 1024x768 (768/frame) | 16,896 | 28,416 | 55,296 | 66,816 | **78,336** | 82,176 |
| 832x768 (624/frame) | 13,728 | 23,088 | 44,928 | 54,288 | **63,648** | 66,768 |

**345 is the column to read.** It is `LONG_LENGTH`, and all 34 shipped API
graphs carry it — verified by reading their `length` widgets, not assumed.
1344x768 is the t2v and image-reference canvas; 1024x768 is what every
video-reference arm ships (`REF_VIDEO_CANVAS`).

**362 is not a legal length.** The column is kept only so old numbers can be
read, and it is the reason to distrust them. `h3_rules.py` applies the
reference's 15.0 s ceiling *after* the frame-count snap, so 362 is 15.083 s
and refused, and 345 is the largest count on the 17n+5 grid. A 362-frame
render still succeeds and nothing says the model is out of distribution,
which is how Run 1 of the 2026-08-14 bench came to be taken there.
`bench_e2e_h3.py` now warns (`34b42b3`).

Upstream's ~100k model ceiling is the 1344x768 / 345-frame corner. The floor
for seeing anything is ~60k tokens; `bench_e2e_h3.py` warns below it.
**A run under the floor produces a null result that reads as "this knob does
nothing".**

Note this is a *token* floor, not a frame floor — 250 frames is 72,576 tokens
at 1344x768 but only 44,928 at 832x768, on opposite sides of the line. Two
shipped graphs stay under it even at 345: `h3_probe_square_canvas` (768x768,
58,752) and `h3_probe_turbo_home_canvas` (960x544, 52,020). Neither enables
Sol-Attn today, and enabling it on either would measure nothing.

---

## The CUDA node

`SolAttnMiniMax`, driving `comfy_kitchen.sol_attn`.

### Options

Four have no Triton counterpart; three Triton options are gone, because the
CUDA kernel routes in INT8 unconditionally and there is no quantization choice
left to make.

**Rows are in widget order, and that is not cosmetic.** A saved graph stores
`widgets_values` as a bare list matched by index, so this table doubles as
that list — regroup it semantically and you will pair a value with the wrong
knob.

Widget order is not quite the declared input order. `tau_profile` is
`force_input`, so it is a socket rather than a widget and takes no slot in
`widgets_values`; it is declared between `verbose` and `dense_blocks` but sits
last here, and the graphs bake **12** values, not 13. Checked against both the
node source and a live `/object_info` — the two disagree in presentation
(`/object_info` reports required inputs before optional ones), which is its
own way to get this wrong.

| option | default | what it does |
|---|---|---|
| `tau` | 1.3 | Routing threshold in sigmas of the proxy row. A key block is exact when its mean score over the query block clears `tau * sqrt(var)`. Higher is sparser. Upstream densities: 1.0 keeps ~16% exact, 1.5 ~7%, 2.0 ~2.7%. |
| `start_percent` | 0.2 | Dense before this point. **Never measured** — see the step table below, it is badly non-linear. |
| `end_percent` | 0.9 | Dense after this point. Also never measured. |
| `min_tokens` | 12288 | Shorter sequences stay dense. `SOL_RECOMMENDED_CUDA` pins 4096, and **neither value changes anything** — but not for the reason first written here. The 50 DiT calls are at the full packed length and the shortest clip past 5 frames is already S = 7,194, so both thresholds take them; the 2 token-refiner calls are ~311 rows, so both thresholds reject them. The conclusion survives the 52-module correction; the "both select everything" reasoning does not. |
| `sink_conditioning` | `exact_kv_and_rows` | See the reference section — this is the dominant knob at reference load. |
| `morton` | False | Z-order the video tokens so each 64-token block is a compact 3D neighbourhood. Exactly neutral for dense attention. |
| `morton_curve` | `2d_frame` | Z-order within each frame, leaving frame order alone. Correct for H3, whose `FRAME_PER_TOKEN` is `(1,4,4,4,4)`. |
| `centroid_tail` NEW | True | One pooled tail per query block instead of per row, 64x less routing work. Upstream: ~1.4x on the **operation**, **~5–10% end to end**, ~5e-4 cosine. |
| `routed_cap_percent` NEW | 0 | Cap routed blocks as a percent of sequence; 0 is uncapped. Bounds the only workspace term growing with T². Below the actual density it silently degrades routed blocks to their pooled term. |
| `reuse_qkv_memory` NEW | False | Write the output into H3's fused qkv buffer instead of allocating. Upstream: ~1.2 GB at 80k tokens, enough to put attention's peak below the FFN's. Safe for H3, which discards that buffer; leave off for other models. |
| `verbose` | False | Per-shape dispatch logging, once per distinct shape. |
| `dense_blocks` | `""` | Blocks kept fully dense, e.g. `0-2,-1`. First and last are the most approximation-sensitive. |
| `tau_profile` NEW | unset | Per-block tau, `blocks=tau` separated by `;` or newlines. `force_input`, so it needs a node wired to it — which is why the graphs bake 12 widget values and not 13. |

### What has actually been measured on it here

#### End to end, 2026-08-14

First e2e measurement of the CUDA node. 1344x768, **362 frames** (107,856
video tokens, above the ~60k floor), 16 steps, `res_multistep`/`simple`,
2 runs plus a discarded warmup, arms alternating, kernel build
`0.2.31+sol.c04ef20`.

| arm | sampler | vs sage |
|---|---|---|
| sage only | 794.7s | 1.000x |
| **shipped (Sol on)** | **493.4s** | **1.611x** |
| shipped, `centroid_tail=0` | 505.8s | 1.571x |
| shipped, `reuse_qkv_memory=1` | 493.5s | 1.610x |

Run-to-run spread was 0.1% on sage and 0.12% on shipped, so the ordering is
not noise.

**Read the baseline before quoting 1.611x.** That sage arm ran
`mode="auto"` -> `fp8_cuda++`, not the shipped `fp16 (most accurate)`,
because the bench had drifted from `h3_config.py` (see `docs/checks.md`).
fp8 sage is the *fast* kernel, so this ratio **understates** against the
shipped configuration. **The understatement is not clean, though.** This
paragraph used to say that with Sol on sage takes **zero** DiT calls, so the
mode affected the sage-only arm and nothing else. That is the claim retracted
at the top of this page, and it survived here after being fixed in two other
places -- caveat decay, in the exact form `docs/checks.md` describes. Sage
runs 5 of 16 steps in the Sol arm, so the fp8 drift hit **both** arms: 16 of
16 steps in sage-only, 5 of 16 in shipped. Correcting both to fp16 slows
sage-only more, so the direction still understates -- but the magnitude is not
pinned, and a corrected re-baseline is the number to quote.

**`centroid_tail` is worth 2.5%**, not the 5-10% upstream reports e2e
(1.611x against 1.571x). Two consequences: it is not where the CUDA
advantage over Triton comes from, and if upstream makes the toggle
unconditional, little is lost here.

**`reuse_qkv_memory` is unmeasured, not neutral.** It came back identical on
time and VRAM, but the VRAM column at the time was reporting torch-active
bytes rather than device usage and resolved only the resident-weight plateau
-- every arm agreed to the megabyte. The instrument could not have shown a
saving. Re-run pending with a device-level metric.

#### Kernel level

On a 4090, `B=1 T=512 H=4 D=128` bf16 at tau 1.3, against the eager reference
the algorithm's author wrote:

- **The two backends' arithmetic is equivalent.** CUDA 0.999919 from the
  reference, Triton INT8 0.999885, Triton bf16 0.999995.
- **They run different tail modes.** CUDA defaults `centroid_tail=True`; Triton
  is per-row and has no such parameter. Measured by grading each against both
  reference modes, not read off the source.
- **`reuse_qkv_memory` is numerically identical** to the normal entry -- cos
  0.999810 against dense SDPA on a fully-exact control, both to six digits. It
  changes where the output lands, not what it is.

#### The exact branch is all-INT8, and that is worth knowing

`sol_attn_exact.cu` runs `mma_s8` for QK into int32 and `mma_u8s8` for PV --
uint8 P, int8 V -- with fp32 output accumulation and bf16 I/O. There is no
fp16/bf16 MMA anywhere in the Sol kernels and no option to enable one, which
is why `int8_qk`/`int8_pv` do not exist on this node: they are unconditional.

This sits oddly beside sage running `fp16 (most accurate)` on the small calls,
but see the note under Quality before drawing the obvious conclusion -- the
accuracy figure that argument rests on is a synthetic-input number.

### `centroid_tail` may stop being a toggle

Upstream is weighing making it unconditional — "probably should even if it's
technically a bit more lossy", on the grounds that there are too many options
for an average user, with model-specific defaults as the alternative. Reported
from conversation, not a shipped change.

If it lands, `centroid_tail=False` disappears and the A/B that separates the
toggle from the kernel becomes unrunnable. **That experiment has a clock on
it.** `SOL_CUDA_DEFAULTS` pins the value, but pinning cannot survive an input
being removed — passing a key the node no longer declares is an error.

---

## The Triton node

`SolAttnPatch`, from [ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton).

### Its status, stated once, because "is this a distraction" keeps getting asked

**Split runtime from tooling and the answer stops being ambiguous.** Audited
2026-08-16 by reading the code, not this page.

**At runtime it is dead, and that is deliberate.** `SolAttnMiniMax` (CUDA) is
what every graph wires and what the owner runs in everything. **Zero shipped
graphs reference `SolAttnPatch` or `SolAttnBlockProbe`** — verified by grep over
`workflows/*.json`, and `check_sol_kernel.py`'s `no_triton_graphs` case fails
the build if one ever drifts back. Nothing you render touches this pack.

**As tooling it is live, in two places, and uninstalling it breaks both:**

1. **`bench/check_solattn_correctness.py` hard-requires it.**  Not "loses a
   cross-check" — the script calls `load_triton_kernels()` up front and
   `return 2` on failure, *before* the CUDA arm (its step 6) is ever reached.
   So removing the pack turns **the only independent correctness check on the
   CUDA Sol kernel** into a permanent skip. And exit 2 in this repo "reads
   exactly like a check that passed", which is the precise failure this pack's
   absence would produce. `docs/checks.md` lists its needs as "CUDA, Triton, and
   a fork build of comfy_kitchen".
2. **`SolAttnBlockProbe` has no CUDA equivalent.** It computes every attention
   call both sparse and dense and logs per-block relative error worst-first —
   the instrument for choosing a `dense_blocks` list. That is not hypothetical:
   `SOL_ARTIFACT_INSURANCE = dict(tau=1.3, dense_blocks="33-35,39-42")` sits in
   `h3_config.py` deliberately unwired, **pending a probe run that has never
   happened**, and `dense_blocks` is the stated fix for the object-dissolve
   artifact under Quality.

**Third role, and this one really is only historical:** `bench_e2e_h3.py
--sol-backend triton` reproduces pre-2026-08-14 numbers. If that were the only
role, the pack would be a distraction.

**Unverified, and it matters to whoever runs the probe:** whether
`SolAttnBlockProbe` works downstream of the *CUDA* node. It wraps
`optimized_attention_override`, and the CUDA node also object-patches the 50 DiT
forwards, so the probe may see none of the real calls — the same shape as the
`ModelAttentionBackend` trap under Ordering. Assume it pairs with the Triton
patch node until someone checks. The resulting block list transfers either way:
`dense_blocks` names model blocks 0-49, not anything kernel-specific.

### Knobs

Its knob set differs: it has `int8_qk`, `int8_pv` and `use_tma`, and lacks
`centroid_tail`, `routed_cap_percent` and `reuse_qkv_memory`. `SOL_RECOMMENDED`
and `SOL_BASELINE_124F` in `h3_config.py` are both written in this vocabulary.

**`int8_qk` selects a different kernel, it does not toggle a dtype.** Read from
`coderef/ComfyUI-SolAttn_triton/__init__.py:214`: `kernel = _sol_attn_int8_kernel if
int8_qk else _sol_attn_kernel`, logged as `int8` or `bf16` (`:222`). So a
bf16-arm against an int8-arm varies the PV dtype, the QK dtype **and** the
implementation at once, and cannot price any one of them. `int8_pv` is passed
only when `int8_qk` is on (`:213`) and **defaults to `True` on the node**
(`:468`) — `SOL_BASELINE_124F` pins it `False`, which is the only reason the
bench's `sage+sol+int8qk` arm is int8-QK with bf16-PV rather than full int8.

### The frontier, at 362 frames

Above the token floor, so these hold. Sampler time, same seed, warmup
discarded, arms alternating, 1344x768, 20 steps:

| arm | sampler | vs sage |
|---|---|---|
| sage only | 991.0 s | 1.00x |
| sol, no int8, tau 1.2 | 827.9 s | 1.20x |
| sol + `int8_qk` + `int8_pv`, tau 1.2 | 714.9 s | 1.39x |
| + tau 1.6 | 648.2 s | 1.53x |
| + tau 2.0 | 602.9 s | 1.64x |

Sparsity and int8 are separate levers of comparable size. **Do not read the tau
1.6 and 2.0 rows as recommendations** — both sit above where the artifact under
Quality appears. They are kept because the cost of each quality knob is only
legible against them.

### The frontier, at 124 frames — BELOW THE FLOOR

Kept for the record. 20 steps:

| config | sampler | vs nothing | vs sage |
|---|---|---|---|
| no sage | 342.1 s | 1.00x | — |
| **sage** | **178.7 s** | **1.91x** | 1.00x |
| sage + sol (`int8_qk`) | 155.0 s | 2.21x | 1.15x |

At the time this was read as "15% is a floor, and length is why", with attention
at ~50% of the step at 124 frames rising to ~76% at 362. That reasoning was
right and the 362-frame table confirmed it. But the measurement itself sits at
37,296 tokens, under the ~60k floor, so it should not be used as a baseline for
anything new.

These arms ran against a Sol-Attn build from 2026-08-05 whose commit was not
recorded — see the process note below.

### Where the time goes

Profiled one forward, 50 DiT blocks, device time only:

| | sage dense | sol sparse (`int8_qk`) |
|---|---|---|
| attention kernel, all 50 blocks | 4296 ms | **2668 ms** |
| per block | 85.9 ms | **53.4 ms** |
| sol routing + quant prep | — | 161 ms (2.1%) |
| everything else | 4944 ms | 4935 ms |
| whole forward | 9240 ms | 7603 ms |

**Non-attention time matches to 0.2%**, which is the control: only attention
changed, so the patching surface is clean.

---

## The unit trap: kernel speed is not end-to-end speed

Upstream reports **CUDA at 1.4x over Triton, end to end**, and separately that
**`centroid_tail` is worth ~5–10% end to end**. The `centroid_tail` tooltip's
"~1.4x faster" is the *operation*.

Pairing those two 1.4x figures — as this document did briefly on 2026-08-14 —
is numerology across different denominators. When only the kernel changes, e2e
speedup can never exceed kernel speedup, so a 1.4x *e2e* win demands a kernel
gap considerably larger than 1.4x; a knob worth 1.4x on the op cannot produce
it, and upstream's own 5–10% figure confirms it does not.

**Never quote a number from `bench_minimax_attn.py` against one from
`bench_e2e_h3.py`.** Attention is only part of a step, and Sol runs only inside
the sigma window on top of that.

---

## The sigma window is not a step fraction

`start_percent` and `end_percent` pass through `percent_to_sigma`, which for a
flow model is `time_snr_shift(shift, 1 - percent)`. H3 runs `shift_video=12.0`,
which bends the curve hard. Computed for `simple`:

| start (end=0.9) | 16 steps | 20 steps |
|---|---|---|
| 0.0 | 15/16 sparse (94%) | 19/20 (95%) |
| 0.1 | 13/16 (81%) | 17/20 (85%) |
| **0.2 (shipped)** | **11/16 (69%)** | **14/20 (70%)** |
| 0.3 | 10/16 (62%) | 13/20 (65%) |
| 0.4 | 8/16 (50%) | 11/20 (55%) |

At 20 steps, `0.2 / 0.9` gives 6 dense steps — 5 leading, 1 trailing — which is
exactly the "simple's 6" recorded when `beta57` was dropped for putting 10 steps
dense instead. Independent path to the same number.

The shape that matters: **0.2 → 0.3 costs one step out of 16.** Not linear.
`end_percent` 0.9 → 1.0 buys exactly one step.

This is also why the scheduler is pinned to `simple`: a different scheduler puts
a different number of steps inside the same percent band, so a scheduler A/B
that does not account for it is measuring two things at once.

---

## References are pinned exact, and a video reference dominates

`_sink_blocks` covers rows `[0, video_start)`. From `PackedLayout` that is text,
keyframe `cond`, keyframe `cond_audio`, `ref_img`, `ref_audio` **and the target
audio segment** — every reference type, whatever it is.

Row counts differ by two orders of magnitude between types (measured, see
`docs/h3_references.md`, at 1344x768):

| reference | DiT rows | also adds to text |
|---|---|---|
| audio, per second | 80 | — |
| image at `match` | ~1,008 | — |
| image at `max`, 1024x1024 | 4,096 | +4,096 |
| image at `max`, 1280x720 | 7,296 | +7,296 |
| video, 960x544, 124 frames | 18,870 | ~+1,700 |
| video, 960x544, 345 frames | **52,020** | +4,667 |

Images and videos pay in **two** segments, and the text segment is inside the
sink as well, so a reference grows the exact region twice.

Share of attention forced exact at a 362-frame 1344x768 target (arithmetic over
those row counts; `S` = sink rows, `T = S + V`, exact work is `T*S` for
`exact_kv` and `2*T*S - S²` for `exact_kv_and_rows`):

| configuration | `exact_kv` | `exact_kv_and_rows` |
|---|---|---|
| t2v, no references | 1.5% | 2.9% |
| 3 images at `match` | 4.1% | 8.1% |
| 3 images at `max` 1280x720 | 29.6% | 50.5% |
| 1 video ref, 124 frames | 17.1% | 31.2% |
| 1 video ref, 345 frames | **35.1%** | **57.9%** |

**"More context" is not "more sparsity."** At the shipped
`exact_kv_and_rows`, one long video reference forces 58% of attention exact and
leaves Sol 42% at any tau — and that is exactly the workload with the most
reason to want Sol.

For heavy-reference work, `sink_conditioning` is the dominant knob, not `tau`
and not `start_percent`: a 23-point swing where `start_percent` 0.2 → 0.3 is one
step of 16.

### The sink conflates two things worth separating

`final_layer(h, t_emb, video_seg, audio_seg)` reads out only the target video
and target audio segments. So the sink holds:

- **reference and conditioning rows** — outputs discarded. They matter only as
  keys and values for later layers, so running *their queries* dense is
  second-order.
- **the target audio segment** — decoded output, first-order, and the stated
  reason `exact_kv_and_rows` is on at all.

`exact_kv_and_rows` treats both identically. Target audio is a single contiguous
segment immediately before video, so `sink_q = [audio_start // BLOCK,
video_start // BLOCK]` would run only the audio queries dense. The kernel takes
an arbitrary `sink_q`; the node hardcodes `sink_q = sink_blocks`. A proposal,
not a measurement, and upstream's call.

---

## Quality

**REOPENED.** Everything in this section was judged from still frames, and the
failure mode that matters is temporal. Read the next subsection first.

### There is no usable fp8-vs-fp16 sage accuracy figure, and the ratios are gone

**Withdrawn 2026-08-16 by the owner, as untrusted.** This subsection used to
carry two competing accuracy ratios for sage's fp8 against its fp16 PV -- one
from a synthetic `torch.randn` sweep run here, one smaller figure reported
secondhand from the sage fork's captured activations -- along with the
`mean_rtol` values behind both. Every one of those numbers has been removed
from this repo rather than caveated. What ruled them out:

- The sweep is `torch.randn`, which is not the input distribution H3 has. On
  iid gaussian input softmax is near uniform, so the output is a
  near-cancelling average and an element-wise relative error is dominated by
  cancellation. The instrument answers a different question than the one asked.
- The real-activation figure was **never re-derived here**, and the script
  producing it is not committed in the sage fork -- the numbers lived in prose
  and commit messages. That is an uncommitted ad-hoc run cited across a repo
  boundary, which is the weakest provenance any claim here had.
- Nothing in `bench/` uses captured activations, so a fresh run today would
  reproduce the synthetic instrument, not replace it.

What survives, because it never depended on a ratio:

- **The fp8 and fp16 sage kernels differ in *both* PV operands** -- P is
  unscaled e4m3 against fp16 -- and no sage kernel exists with mixed operands,
  so nothing isolates V from P. Any claim of the form "quantizing V is the
  lever" over-attributes. What differs between those kernels is an 8-bit PV
  matmul against a 16-bit one.
- **INT8-V specifically is unmeasured** on either side. Sage has no int8-V
  kernel at all.
- A cosine and an rtol from different harnesses cannot be compared; they fail
  in opposite directions.

So the honest statement about Sol's INT8 PV is that it is the same *class* as
sage's fp8 arm, that nobody has measured what that class costs on real H3
activations, and that nobody has measured int8-V. **Not** "Sol throws away the
accuracy we pay 1.58x for" -- that sentence was built on the withdrawn figures.

The captures that would settle it exist
(`~/Storage/h3_captures/2026-08-15_dense_124f_1344x768/`) and no kernel has been
graded against them. Until that runs, this repo has no accuracy figure to quote
and should not acquire one by inference.

### The artifact stills cannot show

At `tau` above roughly 1.5, a small persistent object can dissolve partway
through a clip and be replaced by something else. In frames examined at 24 fps
the transition took about four frames — solid, coming apart, gone — and it never
recovered. Not warping or blurring: the object loses its identity and the model
generates a locally plausible substitute.

**A grid of stills at sampled shot-times cannot catch this**, which is what the
judgements below used. Pick four moments in an eight-second clip and the odds of
landing on both sides of a four-frame transition, on the one small object that
failed, are poor. A gate for this tracks one small persistent object — a hair
ornament, jewellery, on-screen text — frame by frame.

**The fix is to force something dense.** `dense_blocks` exempts the most
approximation-sensitive transformer blocks; `sink_conditioning` does the same
for the packed conditioning rows. Both are cheaper than backing `tau` off far
enough to avoid the problem globally.

`SolAttnBlockProbe` (Triton only) is the instrument for choosing the block list.
Run it at the `tau` you actually intend to use — a profile taken at a gentler
setting is measured where the failure does not occur. `SOL_RECOMMENDED` ships
`dense_blocks=""`. A starting set exists as `SOL_ARTIFACT_INSURANCE`, which is
deliberately not what the graphs wire, pending our own probe run.

### The earlier judgements, at length 124 — below the floor

Treat Sol-Attn as a speed knob. No quality difference held up.

| seed | arm | blind | verdict |
|---|---|---|---|
| 801 | sol+morton+int8qk | no | Sol-Attn better |
| 701 | plain sol | yes | different, neither better |
| 702 | plain sol | yes | nearly identical |
| 1001 | sol+morton+int8qk | no | same |

One positive in four, and it was the first pair looked at. Limits, which are
severe: one observer, one prompt, four pairs; three judgements came late in a
long session, and fatigue pushes toward "these look the same", which is the
direction the nulls point; the prompt is slow-camera, diffuse fog, ambient audio,
where a block-sparse artifact is least likely to surface.

For scale on the noise floor: the same observer called one plain-sage render
"dramatically more interesting" than two others differing only by seed.

### Audio

**RETRACTED: "Sol-Attn renders are consistently louder."** It does not survive
362 frames, where the two mildest configurations measure *quieter* than sage.
What tracks loudness is how aggressive the sparsity is, not whether Sol-Attn is
on. The original was a config effect read as a Sol-Attn effect, from three pairs
at one setting, on an unrecorded build, with morton on — and upstream fixed a
Morton corruption at certain sizes in that window.

What replaces it: audio deviation tracks `tau` (about +1.6 dB mean at 2.0), a
real change nobody has been able to hear. H3's audio rows are ~250–400 in a ~38k
sequence — thin enough to be exactly what a block-sparse router drops first,
the same shape as the object-dissolve artifact. Forcing them dense is the fix,
which is why `sink_conditioning="exact_kv_and_rows"` is the default here and
upstream.

The original measurement, kept for the record:

| pair | seed | mean dB (sage → sol) | peak dB (sage → sol) |
|---|---|---|---|
| 00031/00032 | 1001 | -40.2 → -39.6 (+0.6) | -25.6 → -24.6 (+1.0) |
| 00033/00034 | 1002 | -39.6 → -38.9 (+0.7) | -24.8 → -23.5 (+1.3) |
| 00035/00036 | 1003 | -36.9 → -35.2 (+1.7) | -20.8 → -14.8 (+6.0) |

A mechanism was written to explain that loudness — sparse attention drops
low-weight blocks and renormalizes, so the result leans harder on the strongest
matches. The loudness is the part that did not replicate, so treat the mechanism
as an unsupported story rather than a finding.

Still never checked: accuracy against the sage output rather than loudness. The
files exist, so that is a listening test, not a render.

---

## Configuration findings

Carried over from the Triton evaluation. Everything marked 124 frames is below
the token floor.

- **`tau`: stay at or below ~1.3.** Above roughly 1.5 the object-dissolve
  artifact appears. Costs 82.3 s against tau 2.0 at 362 frames (712.1 vs
  629.8), and worth it.
- **`int8_qk` and `int8_pv` both on** (Triton only). Worth 1.16x on top of plain
  sparsity at 362 frames. Upstream's tooltip says int8 helps at `tau<=1.5` and
  becomes a net loss at `tau>=2.0`.
- **Morton off.** Worth 1.16x alone but a net loss stacked on int8 (1.34x
  against 1.39x). Its arm runs at 94% GPU utilisation where every other arm hits
  99% — the permutation adds non-tensor-core work that stops paying once int8
  has shrunk the exact branch. Upstream now defaults it off too.
- **`sink_conditioning="exact_kv_and_rows"`, on** — but see the reference
  section above, which is where this gets expensive.
- **`dense_blocks`**, from a `SolAttnBlockProbe` run at your own tau. Seven of
  fifty blocks costs roughly 54 s on a 362-frame render. `SOL_RECOMMENDED` ships
  it empty: it does not fix the artifact that tau fixes, and costs 39.2 s.

### Record the commit with every measurement

A process finding, not a configuration one. Sol-Attn shipped five commits on
Aug 4 alone, at least two behaviour-changing at long-clip sizes. The timing arms
and audio numbers on this page were taken without recording which build produced
them, which puts an asterisk on all of them. Snapshot the node's commit
alongside torch and triton versions, or the numbers cannot be defended later.

For the CUDA path this is worse, not better: the branch rebases, and the build
declares a version identical to the stock wheel. `check_sol_kernel.py` prints
what is installed — put it in the run log.

---

## Ordering

```
Load Diffusion Model -> MiniMax H3 SageAttention -> SolAttn* -> SageChainAssert -> BasicGuider
```

Sol must come second: it walks the model's existing object patches and composes
with the attention forwards it finds. Reversed, it overwrites sage's patch and
you silently get sage only.

`SageChainAssert` comes last because it can only grade patches that are already
installed. Every shipped graph wires it there; the diagram above omitted it
until 2026-08-14 while all 71 graphs carried it.

They **alternate rather than stack.** Inside the sigma window Sol runs sparse
and sage is bypassed; outside it, sage runs dense.

**`ModelAttentionBackend` must not be downstream of either.** ComfyUI's
`set_model_optimized_attention` (`comfy/model_patcher.py`) assigns
`optimized_attention_override` unconditionally and its closure discards the
`func` argument — no chaining. Downstream of Sol it deletes Sol silently; the
graph still renders. Sage partly survives because it also object-patches the 50
attention forwards, a path that never reaches `optimized_attention`.
**Re-verified 2026-08-14 against ComfyUI `55b6a9b1`**, after the comfy-kitchen
attention merge, because that merge touched this exact function.

That merge (`bf4c9a08`, #15479) added a second and stricter bypass. When the
node selects "comfy kitchen attention", `set_model_optimized_attention` copies
the backend's `container_function` onto the override and `wrap_attn` dispatches
straight to it — `func` is not even in that signature. "pytorch attention"
leaves `container_function` None and takes the old path. H3 is on the container
path, not exempt from it: `comfy/ldm/minimax/model.py` wraps q/k/v in
`AttentionTensorContainer` before the call. Our overrides declare no
`container_function`, so `wrap_attn` takes the containers and hands us raw
tensors with `func` first — unchanged and still working, but it forfeits
comfy-kitchen's prequantize entry, which only runs when the containers arrive
intact.

**Against a forward-patching sage node the backend is nearly inert, and that is
its own trap.** KJNodes' node and ours both replace
`diffusion_model.blocks.{i}.attn.forward` and so delete the call site; the
comfy-kitchen override reaches none of the 50 DiT blocks, **in either order**,
because the two nodes use different seams and never contend. What it does reach
is the 2 token-refiner blocks, and nothing else. Selecting it on a sage graph
changes 2 of 52 attention modules, which from the outside is indistinguishable
from the backend being slow.

**Sol is what actually breaks, and the combination is worse than either alone.**
`_compose_module_patch` gates on `transformer_options["sol_compose"]`, which
survives the node's `model.clone()`. So inside the sigma window the composed
forward still fires and calls `stock()` — routing the call away from sage's
patch and into comfy-kitchen int8 rather than into Sol's sparse kernel. No Sol
sparsity, no sage patch, dense comfy-kitchen int8, and no error anywhere. Read
from source 2026-08-14, not rendered.

**KJNodes' `MiniMaxH3MemoryEfficientSageAttentionPatch` silently replaces our
sage node, and no check catches it.** Read from source 2026-08-14, not
rendered. It writes `diffusion_model.blocks.{i}.attn.forward` unconditionally
(`nodes/ltxv_nodes.py`); so does ours (`nodes.py`). Pure
last-node-wins, in both directions, with no marker convention. On sm89 its
forward hardcodes `per_channel_fp8(v)` into
`qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf` — fp8 PV, with no mode
input to change it. That is the opposite of the fp16-PV choice `SAGE_NODE`
exists to pin, arrived at without touching a widget.

`SageChainAssert` reports green through it. `require_forward_patch` only tests
that the key list is non-empty (`assert_chain.py`), and their node
never touches `optimized_attention_override`, so ours survives and
`exercise=True` probes *our* override while every real DiT call runs theirs.

**Turning Sol on does not protect you from it.** This page said the opposite
until 2026-08-14 — "with Sol on, sage gets nothing, so only sage-only graphs
bite" — and that was wrong for a reason worth keeping: it reasoned from
`min_tokens` alone and forgot the sigma window. H3's DiT does route every
`optimized_attention` call through one source line
(`comfy/ldm/minimax/model.py`, verified), but Sol only *takes* those calls
inside its window. `_compose_module_patch` declines
on sigma before delegating (`vendor/sol_attn_minimax.py`), and
`make_override` applies the same gate again. At the shipped `0.2 / 0.9`, 16
steps, `shift_video=12.0`, that is **11 sparse and 5 dense** — the same 11/16
this page's own sigma-window table gives, arrived at independently.

So a Sol graph runs sage on 5 of 16 steps, and with their node downstream
those 5 run fp8. They are steps 0-3 and 15: the warm-up that sets structure
and the final refinement. **Whether those steps are more precision-sensitive
than the middle is an inference, not a measurement** — nobody has measured
`start_percent` at any length, which is the same gap flagged at the top of
this page. The exposure is real either way; its cost is not known.

KJNodes' *other* attention node, `MiniMaxLowVRAMAttention`, yields politely —
`if attn_key in m.object_patches: continue` (`nodes/minimax_nodes.py`).
One of their two composes and one wins; only the polite one is written up in
`docs/checks.md`.

No shipped graph wires it. This is a "do not reach for it for VRAM headroom"
note, and an argument for `SageChainAssert` learning to check that the patch
is *ours* — which is the only thing here that would catch it, since the fp8
swap is silent at every other layer.

Confirmed engaged rather than assumed, from the log:

```
[h3] ... sage routed a 2048-token probe on fp16_cuda
[sol_attn] chaining onto an existing attention override
[sol_attn] sparse (1, 37826, 56, 128) tau=1.3
```

The middle line is the ordering check. `smoke_h3.py --log` checks all three.

---

## Measurement traps hit while producing this

Each one produced a confident, wrong number first.

- **ComfyUI caches node outputs.** Re-submitting an identical graph executes
  nothing and returns in milliseconds. The first e2e bench reused one seed and
  reported a **405x speedup** for a render that never ran.
- **Run 1 of any Triton arm pays autotune.** Comparing a warm arm against a cold
  one made Morton look like it helped by 3 points. It does not.
- **Reasoning from wall-clock is not profiling.** Per-step arithmetic suggested
  routing overhead was eating the benefit. The profile put routing at 2.1% and
  the real answer at Amdahl. The arithmetic was fine and the conclusion wrong.
- **Comparing finished renders numerically measures chaos, not quality.** At 20
  steps of a flow-matching ODE any perturbation diverges the trajectory, so
  same-seed latents differ substantially while both look fine.
- **Synthetic inputs cannot answer questions about input distribution.** A
  `torch.randn` sweep reported `smooth_k` as having no effect, which it would
  have reported whether or not the effect were real.
- **Grading a kernel against the wrong tail mode invents a winner.** Costs about
  1.2e-3 of apparent accuracy — larger than the gap between the backends. See
  `docs/checks.md`.
- **An arm below the token floor produces a null that reads as a finding.**

---

## Reproducing

```bash
# CUDA, above the token floor
python bench/bench_e2e_h3.py --arms sage,sage+sol --runs 2 --length 362

# the centroid_tail A/B — the one experiment with a deadline on it
python bench/bench_e2e_h3.py --sol-backend cuda --length 362 \
    --arms 'sage+sol,sage+sol[centroid_tail=0]' --runs 2

# start_percent, never measured
python bench/bench_e2e_h3.py --length 362 --runs 2 \
    --arms shipped,shipped+start0.0,shipped+start0.1,shipped+start0.3,shipped+start0.4

# reproduce a pre-2026-08-14 number
python bench/bench_e2e_h3.py --sol-backend triton --arms sage,sage+sol+int8 --length 362
```

`--sol-backend` defaults to `cuda`. Named arms are written in the Triton
vocabulary, so ones using `int8_qk`/`int8_pv` refuse under `cuda` rather than
silently dropping the knob and becoming a different arm.

A render at 362 frames is roughly 12 minutes, so a 5-point sweep at 2 runs is
about 2 hours.

---

## What is open

| question | why it matters | blocker |
|---|---|---|
| `centroid_tail` on/off, e2e | separates the toggle from the kernel | **none — has a deadline, upstream may remove the toggle** |
| `sink_conditioning` at reference load | 23-point swing, biggest lever there is | bench is t2v-only, needs reference wiring |
| `start_percent` 0.0–0.4 | zero measurements, ever | none, arms exist |
| `min_tokens` 4096 vs 12288 | our pin is a third of the node's crossover | none |
| re-baseline the frontier above 60k tokens | most numbers here are the wrong regime | GPU hours |
| CUDA e2e vs Triton e2e, ours | we have upstream's 1.4x, not our own | none |
| quality at tau 1.3, watched to the end | the artifact is temporal and length-dependent | a human watching |
