# Sol-Attn on MiniMax H3

Sol-Attn ([arXiv 2607.24027](https://arxiv.org/abs/2607.24027)) is training-free
block-sparse attention. Each 64-token query block attends a routed subset of key
blocks exactly and covers the rest with one pooled term per block, so the whole
sequence still contributes to the softmax denominator.

**Two implementations exist and this page covers both.** As of 2026-08-14 the
CUDA one is what every shipped graph wires and what this repo measures against.
The Triton one is what every number older than that date was taken on, and it
stays installed for that reason. They are not interchangeable and they do not
share a knob vocabulary — see `SOL_RECOMMENDED_CUDA` in `h3_config.py` for what
the migration did and did not carry over.

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
| every e2e timing from Run 1, incl. **1.611x** | taken at **362 frames, an illegal length** (15.083s, past the 15.0s ceiling), against an **fp8** sage baseline the graphs do not ship, and before it was known that sage runs 5/16 steps in the Sol arm. Wrong on three axes. |
| `centroid_tail` = **2.5%** | the measurement is sound (two runs, 0.1% spread) but it is at 362. Ordering is trustworthy; the magnitude is not pinned to a shipped config. |
| `reuse_qkv_memory` "no VRAM saving" | **uninformative, not negative.** The peak-VRAM column was reporting torch-active bytes, which resolved only the resident-weight plateau. The instrument could not have shown a saving. |
| the **2.7x** fp8-vs-fp16 accuracy figure | synthetic `torch.randn` input; **1.3x** on real captured H3 activations. Everything in `bench/` inherits this -- `bench_minimax_attn.py:201` builds `randn` and nothing here uses captured activations. |
| **0.999919** and the other correctness cosines | **implementation fidelity, not accuracy.** Kernel is graded against the reference *at the same tau*, so the sparse approximation is on both sides and cancels. Not comparable to any dense kernel's accuracy number. Also T=512 synthetic. |
| **1.4x** CUDA over Triton e2e | upstream conversation. Never reproduced here. |
| `centroid_tail` **5-10%** e2e | upstream conversation. Ours measured 2.5%. |
| "38 text rows", "sequence 12,264" | from `smoke_h3.py`, which **substitutes both the prompt and the length**. Not a scaled-down version of any shipped graph. Everything derived from it -- that audio dominates the sink, that text is the whole v2 narrowing, that `sink_q` start is 0 -- is a statement about the harness. |
| the reference-load table's **35.1% / 57.9%** | wrong on three axes for what we ship: a **v1** formula (v2 changed the mechanism), at **362**, at **1344x768** where the ref graphs are 1024x768. |
| "with Sol on, sage gets nothing" | **retracted.** Sage runs **5 of 16 steps** -- the sigma window, not just `min_tokens`. |
| bench progress read from its own stdout mid-run | the warmup `print` lacks `flush=True`; under `tee` stdout is block-buffered. A finished render can read as "still in warmup" for 20 minutes. Read ComfyUI's log instead. |

### Can rely on

All verified on this box today, by running rather than reading:

- **The CUDA seam works.** Live render, `cuda-int8` in the log, sage's override found and chained, 50 forwards composed.
- **345 is legal, 362 is not.** `h3_rules.py:25` and diffusers `before_denoise.py:399-407`, independently.
- **H3's DiT has exactly one attention site**, `comfy/ldm/minimax/model.py:184`, at the full packed length.
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
| pack | `custom_nodes/ComfyUI-SolAttn-cuda/` | `custom_nodes/ComfyUI-SolAttn_triton/` |
| needs | a source build of `comfy_kitchen`'s `sol_attn` branch | nothing beyond Triton |
| speed | upstream reports **1.4x over Triton at the same tau, end to end** | baseline |
| accuracy vs the algorithm's reference | 0.999919 | 0.999885 (int8), 0.999995 (bf16) |
| default tail mode | `centroid_tail=True` | per-row, not adjustable |
| status here | **what every shipped graph wires, and what new work measures** | kept for reproducing pre-2026-08-14 numbers |

**Use CUDA.** The backends are arithmetically equivalent (see below), so there
is no accuracy argument for Triton, and CUDA is faster. Triton stays installed
because pre-2026-08-14 numbers were taken on it and because
`bench/check_solattn_correctness.py` grades the two against one shared oracle —
a cross-check that only exists while both are present.

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
| `min_tokens` | 12288 | Shorter sequences stay dense. `SOL_RECOMMENDED_CUDA` pins 4096, and **neither value changes anything**: H3's DiT has one attention site at the full packed length, and the shortest clip past 5 frames is already S = 7,194. Both thresholds select everything. |
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
shipped configuration. And the understatement is one-sided: H3's DiT has
exactly one attention site (`comfy/ldm/minimax/model.py:184`) at the full
packed length, so with Sol on, sage takes **zero** DiT calls -- the mode
affects the sage-only arm and nothing else. A corrected re-baseline is the
number to quote.

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
Shipped graphs wired this until 2026-08-14; they now wire the CUDA node. It
stays installed for reproducing older numbers and for `SolAttnBlockProbe`.

Its knob set differs: it has `int8_qk`, `int8_pv` and `use_tma`, and lacks
`centroid_tail`, `routed_cap_percent` and `reuse_qkv_memory`. `SOL_RECOMMENDED`
and `SOL_BASELINE_124F` in `h3_config.py` are both written in this vocabulary.

It also carries `SolAttnBlockProbe`, which computes every attention call both
sparse and dense and logs per-block relative error worst-first. That is the
instrument for choosing a `dense_blocks` list, and it has no CUDA equivalent —
the one live reason to reach for Triton.

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

### The 2.7x sage accuracy figure is synthetic, and real activations say 1.3x

Relevant here because Sol's exact branch is all-INT8 (uint8 P, int8 V), which
looks damning next to this repo running sage at `fp16 (most accurate)` on the
grounds that 8-bit V costs 2.7x accuracy. **That comparison does not hold up.**

Reported by the sage fork's claude on 2026-08-14, from its CHANGELOG v0.7.0
and commits `13b19e0` / `12a5872` / `1f619b4`; not re-derived here:

- The 2.7x is measured on `torch.randn` q/k/v. On q/k/v **captured from a real
  H3 forward**, fp8++ lands at mean_rtol 0.026 rather than 0.098, and the
  fp8-to-fp16 gap **narrows from 2.6x to 1.3x**. Their own verdict line: "Every
  accuracy figure this repo quotes from a synthetic bench is a pessimistic
  bound, not an estimate."
- The mechanism is worth keeping: on iid gaussian input softmax is near
  uniform, so the output is a near-cancelling average of S random vectors.
  Elements sit near zero, and an element-wise relative error with a symmetric
  denominator is dominated by cancellation. Cosine is blind to exactly that.
  The two metrics fail in opposite directions, which is why a cosine and an
  rtol from different harnesses cannot be compared.
- **"Quantizing V to fp8 at all is the lever" over-attributes.** The fp8 and
  fp16 sage kernels differ in *both* PV operands -- P is unscaled e4m3 against
  fp16 -- and no sage kernel exists with mixed operands, so nothing isolates V
  from P. What was measured is an 8-bit PV matmul against a 16-bit one.
- INT8-V specifically is **unmeasured** on either side. Sage has no int8-V
  kernel at all.
- Provenance: the script producing those tables is not committed in the sage
  fork. The numbers live in prose and commit messages. Cite as an uncommitted
  ad-hoc run.

So the honest statement about Sol's INT8 PV is that it is the same *class* as
the arm this project measured as costlier, that the cost of that class on real
H3 activations is 1.3x rather than 2.7x, and that nobody has measured int8-V.
Not "Sol throws away the accuracy we pay 1.58x for".

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
`set_model_optimized_attention` (`comfy/model_patcher.py:688`) assigns
`optimized_attention_override` unconditionally and its closure discards the
`func` argument — no chaining. Downstream of Sol it deletes Sol silently; the
graph still renders. Sage partly survives because it also object-patches the 50
attention forwards, a path that never reaches `optimized_attention`.

**KJNodes' `MiniMaxH3MemoryEfficientSageAttentionPatch` silently replaces our
sage node, and no check catches it.** Read from source 2026-08-14, not
rendered. It writes `diffusion_model.blocks.{i}.attn.forward` unconditionally
(`nodes/ltxv_nodes.py:2194`); so does ours (`nodes.py:137`). Pure
last-node-wins, in both directions, with no marker convention. On sm89 its
forward hardcodes `per_channel_fp8(v)` into
`qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf` — fp8 PV, with no mode
input to change it. That is the opposite of the fp16-PV choice `SAGE_NODE`
exists to pin, arrived at without touching a widget.

`SageChainAssert` reports green through it. `require_forward_patch` only tests
that the key list is non-empty (`assert_chain.py:305-308`), and their node
never touches `optimized_attention_override`, so ours survives and
`exercise=True` probes *our* override while every real DiT call runs theirs.

**Turning Sol on does not protect you from it.** This page said the opposite
until 2026-08-14 — "with Sol on, sage gets nothing, so only sage-only graphs
bite" — and that was wrong for a reason worth keeping: it reasoned from
`min_tokens` alone and forgot the sigma window. H3's DiT does have exactly one
`optimized_attention` site (`comfy/ldm/minimax/model.py:184`, verified), but
Sol only *takes* that call inside its window. `_compose_module_patch` declines
on sigma before delegating (`vendor/sol_attn_minimax.py:571-574`), and
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
`if attn_key in m.object_patches: continue` (`nodes/minimax_nodes.py:193`).
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
