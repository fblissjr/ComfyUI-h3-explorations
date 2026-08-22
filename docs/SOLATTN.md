# Sol-Attn on MiniMax H3

Sol-Attn ([arXiv 2607.24027](https://arxiv.org/abs/2607.24027)) is training-free
block-sparse attention. Each 64-token query block attends a routed subset of key
blocks exactly and covers the rest with one pooled term per block, so the whole
sequence still contributes to the softmax denominator.

## Start here: this page, and the two it links to

**This is the Sol-Attn entry point and the authority.** It owns the knobs, the
sink, the ordering rules, and every Sol-Attn number measured on this box. Three
deep dives hang off it, each owning a topic this page deliberately does not:

| page | owns | do not |
|---|---|---|
| [`docs/morton.md`](morton.md) | token order: block geometry, the curves, the capture analysis, the six-arm ordering sweep, the assumption chain | quote it against this page's config values |
| [`docs/sol_upstream.md`](sol_upstream.md) | what upstream says: the paper, Sol-Engine's per-profile H3 recipes, the other ComfyUI packs | read any number there as comparable to ours |
| [`docs/h3_input_impacts.md`](h3_input_impacts.md) | how canvas, frame count and Sol settings interact: the per-canvas Morton `3d` ranking over all 48 legal canvases, the `latent_t % 4` length effect, the token floor crossed with both axes, and block maps | read its geometry tables as a quality ranking |

The rule that keeps them from drifting, after `docs/SOLATTN.md` and
`docs/morton.md` spent a day asserting opposite Morton figures: **a number is
stated once, in the page that owns it, and everywhere else is a one-line verdict
plus a `Canonical:` link.** If this page and a deep dive disagree about
something the deep dive owns, the deep dive is right.

Repo-wide ledgers sit above all three: [`docs/evidence.md`](evidence.md) for
claims that were measured and should not be relied on,
[`docs/checks.md`](checks.md) for what is guarded,
[`docs/open_experiments.md`](open_experiments.md) for what is deliberately not
measured.

## What the published method actually claims

Read 2026-08-16 at the depth recorded in
[`docs/sol_upstream.md`](sol_upstream.md) -- abstract, ablation summary and
HTML, not the full PDF. Four things matter for how this page is read:

- **The contribution is the correction, not the routing.** Unselected blocks
  reuse their proxy scores rather than being dropped, and the paper's ablation
  shows that advantage **widening as sparsity rises**. So `centroid_tail` is the
  method's core claim, not the side knob this page treated it as.
- **`tau` is the paper's `beta`** in `t_i = mu_i + beta * sigma_i`, confirming
  what the CUDA source says it means -- **and the paper never sweeps it.**
  Nothing upstream adjudicates our 1.3 against Sol-Engine's 1.0.
- **H3 is not evaluated anywhere in the paper.** The 4090 build, the H3 port and
  Morton are all work sitting on top of the published method.
- **No token reordering appears in it.** See [`docs/morton.md`](morton.md).

**Two implementations exist and this page covers both.** As of 2026-08-14 the
CUDA one is what every shipped graph wires and what this repo measures against.
The Triton one is what every number older than that date was taken on, and it
was **DELETED on 2026-08-16** (commit `6872dfd`) after its last two
dependencies were removed. It is recoverable from
`github.com/kijai/ComfyUI-SolAttn_triton` at `842c4ea`; nothing in this tree
references it. This page still covers it because pre-2026-08-14 numbers were
taken on it and they are still quoted here. They are not interchangeable and
they do not share a knob vocabulary —
see `SOL_RECOMMENDED_CUDA` in `h3_config.py` for what the migration did and did
not carry over.

Everything here is single-machine (RTX 4090, sm_89), single-workload. The sage
baseline is [SageAttention-ada](https://github.com/fblissjr/SageAttention-ada),
a fork, never measured against stock SageAttention — so every ratio means "on
this fork, this model, this box".

---

## Read this before quoting any number on this page

Status 2026-08-16. The table below was written 2026-08-14, when this page
changed more in one day than in the two weeks before it. Two things have
happened since: **every fp8-vs-fp16 accuracy ratio was withdrawn** (see the
row below and `docs/evidence.md`), and **the Triton backend was deleted**, so
the Triton numbers here are now unreproducible without recovering the pack
from upstream. A large share of what this page says is
labelled rather than deleted. Read this list first.

### "int8" on this page means three different things

Added 2026-08-17, because a row below gives "stacked on int8" as a retraction
reason and does not say which one it means.

| when a claim says int8 | it may mean | how to tell |
|---|---|---|
| **the attention kernel** | the CUDA Sol kernel, which routes in INT8 unconditionally | `cuda-int8` in the log |
| **the DiT and CLIP weights** | the shipped checkpoints, which `workflows/h3_config.py` names `pruned_int8_convrot` | nothing in a log. The loader prints `dtype: torch.float16` for both builds and cannot distinguish int8 storage from a dequantized fallback |
| **the VAE decoder** | a separate `int8_convrot` build | resident allocation, not a log line |

**The Morton retraction below means the attention kernel** — that reading comes
from its Triton and `tau` context, not from the row, which is why this table
exists.

And a standing condition on everything here, not a footnote: the weights are
**pruned, convrot-rotated and int8**, three separate properties, and
`fp8_scaled` and `w4a8_mixed` builds of the same models sit on disk beside them.
`convrot` applies a rotation (`docs/roadmap.md`, established while grading the
ref LoRA), so anything on this page that reasons about q/k geometry — centroid
similarity, block locality, what a token ordering buys — is reasoning about a
rotated, pruned, quantized space. Whether that reaches the ordering is unmeasured
and filed as an open experiment; nothing here should be read as though it were
settled either way.

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
| Morton **"worth 1.16x alone, 94% GPU utilisation"** | **RETRACTED 2026-08-16.** Triton, 362 frames, stacked on int8. On CUDA the permutation is free -- two control pairs disagreeing in sign, both under the noise floor. It had been sitting in this page's Configuration findings and in `h3_config.py` as a live argument |
| the Morton permutation **"costs 0.8 s of 861, or 1.0009x"** | the replacement for the row above, and wrong the same way in miniature: one arm of a two-arm control quoted as the isolated number, while the sparse pair moved 1.2 s the *other* way. Both are under this bench's run-to-run spread on single runs. **Free is the claim; 1.0009x is not a measurement of anything** |
| **any quality A/B on this page, as a like-for-like** | all of it was taken on `res_multistep`. The default sampler is `er_sde` since 2026-08-15, which injects noise every step. A knob that perturbs attention numerics reads as more "reseeded" under it. Timings should carry (both are one eval per step); quality comparisons are owed a re-run. |

### Can rely on

All verified on this box today, by running rather than reading:

- **The CUDA seam works.** Live render, `cuda-int8` in the log, sage's override found and chained, 50 forwards composed.
- **362 is the max length, 345 is where diffusers stops.** `h3_rules.py` (`MAX_LENGTH`) and diffusers `before_denoise.py`, which answer two different questions -- see `reference_would_emit()`.
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
| pack | `custom_nodes/ComfyUI-SolAttn-cuda/` | **deleted 2026-08-16**; upstream `kijai/ComfyUI-SolAttn_triton@842c4ea` |
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
  pins. Tag your build (this box uses `0.2.31+sol.23d1a66`) or nothing can
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

| canvas | 73 frames | 124 | 250 | 300 | 345 | **362** |
|---|---|---|---|---|---|---|
| 1344x768 (1008/frame) | 22,176 | 37,296 | 72,576 | 87,696 | **102,816** | 107,856 |
| 1024x768 (768/frame) | 16,896 | 28,416 | 55,296 | 66,816 | **78,336** | 82,176 |
| 832x768 (624/frame) | 13,728 | 23,088 | 44,928 | 54,288 | **63,648** | 66,768 |

**362 is the column to read.** It is `LONG_LENGTH` as of 2026-08-16, and all
shipped API graphs carry it — verified by reading their `length` widgets, not
assumed. 1344x768 is the t2v and image-reference canvas; 1024x768 is what
every video-reference arm ships (`REF_VIDEO_CANVAS`).

**The 345 column is history, and it is why some numbers here do not compare.**
345 was the default between 2026-08-10 and 2026-08-16, on the argument that
diffusers would also emit it. Measurements taken in that window are at 345;
everything before and after is at 362. The 5% length difference should not
move a ratio, but it was never re-checked in either direction. 345 remains the
answer to "would diffusers emit this" — `h3_rules.reference_would_emit()` —
and nothing more.

Upstream's ~100k model ceiling is the 1344x768 / 362-frame corner. The floor
for seeing anything is ~60k tokens; `bench_e2e_h3.py` warns below it.
**A run under the floor produces a null result that reads as "this knob does
nothing".**

Note this is a *token* floor, not a frame floor — 250 frames is 72,576 tokens
at 1344x768 but only 44,928 at 832x768, on opposite sides of the line. Two
shipped graphs stay under it even at 362: `h3_probe_square_canvas` (768x768,
58,752 at 345) and `h3_probe_turbo_home_canvas` (960x544, 52,020 at 345). Neither enables
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

Widget order is not quite the declared input order, for two reasons.

`tau_profile` is `force_input`, so it is a socket rather than a widget and
takes no slot in `widgets_values`; it sits last here. Checked against both the
node source and a live `/object_info` — the two disagree in presentation
(`/object_info` reports required inputs before optional ones), which is its
own way to get this wrong.

And since the v3 node (2026-08-22) `selection` is a **DynamicCombo**: picking
an option adds that option's own inputs to the node, so the widget list has a
variable middle. The chosen option's widgets are spliced in **immediately
after the selector**, not appended — source read at ComfyUI_frontend v1.49.6,
`src/core/graph/widgets/dynamicWidgets.ts`. The API form spells the same thing
differently, keying them under the combo with a dot (`selection.tau`), which
ComfyUI regroups into the dict the node receives.

**Do not hand-edit a Sol node in a saved graph.** A graph carrying the old
pre-v3 inputs passes ComfyUI's prompt validation and then dies at execute, so
the queue is the first thing that tells you. `workflows/build_workflows.py`
owns both spellings; regenerate.

| option | default | what it does |
|---|---|---|
| `selection` NEW | `adaptive tau` | Which rule picks the exact key blocks. `adaptive tau` is the threshold every number on this page was measured under and what every graph here ships. `top-k (SLA)` is the other option and brings `keep_percent` instead of `tau`. |
| `keep_percent` NEW | 10.0 | Only under `top-k (SLA)`. Percent of key blocks each query block keeps exactly, a fixed density everywhere rather than one that varies per head and block; sinks and the diagonal still ride on top. **This is not `MiniMaxH3SLARouter`** — read the row below the table before treating the two as arms of one comparison. |
| `tau` | 1.3 | Only under `adaptive tau`. Routing threshold in sigmas of the proxy row. A key block is exact when its mean score over the query block clears `tau * sqrt(var)`. Higher is sparser. Upstream densities: 1.0 keeps ~16% exact, 1.5 ~7%, 2.0 ~2.7%. |
| `start_percent` | 0.2 | Dense before this point. **Never measured** — see the step table below, it is badly non-linear. |
| `end_percent` | 0.9 | Dense after this point. Also never measured. |
| `min_tokens` | 12288 | Shorter sequences stay dense. `SOL_RECOMMENDED_CUDA` pins 4096, and **neither value changes anything** — but not for the reason first written here. The 50 DiT calls are at the full packed length and the shortest clip past 5 frames is already S = 7,194, so both thresholds take them; the 2 token-refiner calls are ~311 rows, so both thresholds reject them. The conclusion survives the 52-module correction; the "both select everything" reasoning does not. |
| `sink_conditioning` | `exact_kv_and_rows` | Keeps the target audio's queries exact. **NOT the dominant knob at reference load** — that was v1 arithmetic; under the v2 node the swing is ~0.5 points, not 23. See the reference section. |
| `morton` | False | Z-order the video tokens so each 64-token block is a compact 3D neighbourhood. Neutral for dense attention **in exact arithmetic** -- not bit-identical, measured. **Under Sol it is not a free toggle: block membership feeds `kcvar`, so turning it on moves the routing threshold and the routed density at a fixed `tau`.** Direction not derivable, unmeasured. `Canonical: docs/morton.md` |
| `morton_curve` | `2d_frame` | Node default. Z-order within each frame, leaving frame order alone. **`SOL_RECOMMENDED_CUDA` pins `3d` since 2026-08-16**, on a centroid-fidelity measurement; changes nothing while `morton=False`. `Canonical: docs/morton.md` |
| `centroid_tail` NEW | True | One pooled tail per query block instead of per row, 64x less routing work. Upstream: ~1.4x on the **operation**, **~5–10% end to end**, ~5e-4 cosine. **Ours measured 2.5% e2e, which makes this the smallest knob in the node, not the largest.** The tooltip's "~1.4x" has been read as end-to-end twice; see `docs/evidence.md`. |
| `reuse_qkv_memory` NEW | False | Write the output into H3's fused qkv buffer instead of allocating. Upstream: ~1.2 GB at 80k tokens, enough to put attention's peak below the FFN's. Safe for H3, which discards that buffer; leave off for other models. |
| `verbose` | False | Per-shape dispatch logging, once per distinct shape. |
| `dense_blocks` | `""` | Blocks kept fully dense, e.g. `0-2,-1`. First and last are the most approximation-sensitive. |
| `tau_profile` NEW | unset | Only under `adaptive tau`. Per-block tau, `blocks=tau` separated by `;` or newlines. `force_input`, so it needs a node wired to it — a socket, not a widget value. |

`routed_cap_percent` was here until 2026-08-22 and the v3 node does not
declare it. It capped the routed-block list as a percent of the sequence to
bound the one workspace term growing with T².

**`top-k` agrees with the algorithm less closely than `tau` does, measured
2026-08-22.** At B=1 T=1024 H=8, kernel against the vendored oracle with the
same selection on both sides, `tau=1.0` lands at cos 0.9987 while `topk_ratio`
lands at 0.975-0.983 -- the top-k path is roughly an order of magnitude
further from its own reference. Consistent with what the CUDA source says of
its threshold ("matches exact top-k up to int8 rounding at the boundary"),
which the tau path does not have to survive the same way. `bench/probe_sol_topk.py`
is the run; **read its caveat before quoting either number** -- synthetic input
gives a near-uniform softmax and nothing for a router to find, so only the
ratio between the two selections at one shape means anything, and neither
absolute figure does.

**`top-k (SLA)` is not the SLA router, and the difference is the tail.** Under
either selection Sol still adds a pooled term for every block it did not pick,
so nothing leaves the softmax. `MiniMaxH3SLARouter` — the arm the Turbo-SLA
LoRA was distilled under — drops unpicked blocks outright. Read in the eager
implementation, where `topk_ratio` changes which blocks are marked exact and
leaves the tail branch alone. So Sol at `top-k` is a third attention, not a
cheaper spelling of the router, and no arm here has been rendered under it.

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

#### End to end, 2026-08-16: a clean Sol-versus-dense number

**Sol-Attn is worth 1.896x on the sampler**: 860.8 s dense against 454.0 s
sparse, 1344x768, 294 frames, 16 steps, one run per arm.

This is the number to quote instead of the retracted 1.611x, and the reason is
the baseline. Both arms are the **same shipped config with only `dense_blocks`
changed**, so the comparison isolates the sparsity and nothing else. The 1.611x
arm compared against an fp8 sage baseline no graph ships, which is wrong in a
direction nobody has pinned.

It came out of the ordering sweep, whose other five arms are about token order
and live in [`docs/morton.md`](morton.md) with the method. Two caveats travel
with it: one run per arm, and 294 frames rather than the shipped 362.

**Do not quote peak VRAM from that run.** Morton appeared to save 3.7 GB across
all three sparse curves; the dense control showed it *costing* 2.1 GB. Opposite
signs, so it is an allocator artifact, not an arm effect.

#### End to end, 2026-08-18: sage mode against Sol reachability, and why the shipped sage default changed

**The data is [`bench/results/2026-08-18_attention_defaults.json`](../bench/results/2026-08-18_attention_defaults.json),
generated by [`bench/results/make_attention_defaults_json.py`](../bench/results/make_attention_defaults_json.py).
Read the ratios there, not here.** They are deliberately not restated in this
prose: a figure copied into a sentence is a second copy, and this page already
carries a retraction table full of numbers that outlived their conditions.

Six arms, one variable each, same graph and seed. The two axes are the sage
kernel mode and whether a `SolAttnMiniMax` node is **reachable from an output
node** — not whether it is present, which is the distinction
[`docs/evidence.md`](evidence.md) records a capture provenance field getting
backwards. Three things came out of it that change what this repo ships:

- **The shipped sage default moved to `auto`**, set in
  [`workflows/h3_config.py`](../workflows/h3_config.py)`::SAGE_NODE` and written
  into every graph by [`workflows/build_workflows.py`](../workflows/build_workflows.py).
  The reason is not the speed, which was always known and always accepted. It is
  that the surviving argument for `fp16 (most accurate)` was a perceptual A/B
  taken at 124 frames with **Sol-Attn absent**, one day before Sol landed. With
  Sol on, sage runs only the steps outside the sigma window, so fp16 was being
  paid for on every step and delivering on a minority of them. The full reasoning
  and what would reverse it are in the `SAGE_NODE` comment.
- **Sol's own per-step cost does not move with the sage mode.** The sparse-step
  figure is the same in both arms, so the entire fp16 difference sits in the
  dense steps. That is why the sigma window and the sage mode interact, and why
  picking the mode first is the right order.
- **An unreachable Sol node is expensive and silent.** A hand-built graph with an
  *active* `SolAttnMiniMax` whose MODEL output feeds nothing renders dense and
  looks entirely normal. `bench/preflight_graph.py` reports this now, on
  UI-format graphs too; see [`docs/checks.md`](checks.md).

Caveats travel in the data file rather than here, because that is the artifact a
later reader will quote. The load-bearing one: the board power limit was not
stock for this run, so the **ratios** hold and the absolute seconds are not
comparable to anything recorded earlier on this box.

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

**Two things found 2026-08-16 make that likelier, and both raise the stakes on
the A/B.** Sol-Engine's own public entry point has **no toggle at all**
(`coderef/Sana/techniques/sparse_backends/sol_attn/interface.py:397-408`), so the reference implementation already treats it as
unconditional. And the paper treats the correction as its core contribution
rather than an optimisation, with an ablation showing its advantage growing as
sparsity rises. A knob this page priced at 2.5% of render time is the thing the
method is *for*. See [`docs/sol_upstream.md`](sol_upstream.md).

If it lands, `centroid_tail=False` disappears and the A/B that separates the
toggle from the kernel becomes unrunnable. **That experiment has a clock on
it.** `SOL_CUDA_DEFAULTS` pins the value, but pinning cannot survive an input
being removed — passing a key the node no longer declares is an error.

---

## The Triton node

`SolAttnPatch`, from [ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton).

### Its status: DELETED 2026-08-16, and what that cost

**The pack is gone from this tree.** Not moved — deleted, commit `6872dfd`.
Recoverable from `github.com/kijai/ComfyUI-SolAttn_triton` at `842c4ea`.
`SolAttnPatch` and `SolAttnBlockProbe` are absent from a live `/object_info`;
verified after a restart.

**This section argued for keeping it until an hour before the deletion, and
that argument is left here rather than removed**, because one leg of it was
resolved and the other was accepted as a cost. Deleting the argument would hide
which.

**The runtime case was never in doubt.** `SolAttnMiniMax` (CUDA) is what every
graph wires and what the owner runs in everything. Zero shipped graphs ever
referenced the Triton nodes, and `check_sol_kernel.py`'s `no_triton_graphs`
case fails the build if one drifts back.

**Dependency 1 — RESOLVED, and it had to be before deletion was safe.**
`bench/check_solattn_correctness.py` used to hard-require the pack: it loaded
the Triton kernels up front and returned 2 on failure, *before* its CUDA arm
was reached. So deleting the pack would have turned the only independent
correctness check on the CUDA kernel into a permanent skip — and exit 2 here
reads exactly like a check that passed. That coupling was an accident of
control flow, not of method: the CUDA arm never needed Triton. The check now
grades CUDA only, its red control was shown red against a copy, and it runs
green with the pack absent.

**Dependency 2 — NOT resolved. This is a real cost and it is open.**
`SolAttnBlockProbe` computed every attention call both sparse and dense and
logged per-block relative error worst-first. It was the instrument for choosing
a `dense_blocks` list, it had **no CUDA equivalent**, and it is now gone from
the tree. Consequences, stated plainly so nobody plans against a tool that is
not here:

- `SOL_ARTIFACT_INSURANCE = dict(tau=1.3, dense_blocks="33-35,39-42")` in
  `h3_config.py` is a **guess**, sitting unwired pending a probe run — and that
  run is now blocked on an instrument that does not exist in this tree.
- `dense_blocks` is the stated fix for the object-dissolve artifact under
  Quality. There is currently no way to choose the list from measurement.
- The replacement is scaffolded and **not implemented**: `sol_block_probe.py`
  in this repo, all stubs, port target `842c4ea:__init__.py:323-342`. Until it
  lands, "run the probe" is not an available action.

**Still unverified, and it decides whether the port is even possible:** whether
a probe wrapping `optimized_attention_override` sees the CUDA node's DiT calls
at all. The CUDA node object-patches the 50 DiT forwards; reading
`_compose_module_patch`, it calls `stock()` inside the sigma window, which
should reach the override. That is an inference, and the Ordering section below
documents two nodes that look like they compose and do not. One render settles
it, and it is the first thing the port needs.

**Third role, genuinely only historical:** `--sol-backend triton` reproduced
pre-2026-08-14 numbers and now refuses at argparse. Every number it could have
reproduced already carries this page's "do not rely on" caveats.

### Knobs

Its knob set differs: it has `int8_qk`, `int8_pv` and `use_tma`, and lacks
`centroid_tail`, `routed_cap_percent` and `reuse_qkv_memory`. `SOL_RECOMMENDED`
and `SOL_BASELINE_124F` in `h3_config.py` are both written in this vocabulary.

**`int8_qk` selects a different kernel, it does not toggle a dtype.** Read from
`kijai/ComfyUI-SolAttn_triton@842c4ea:__init__.py:214`: `kernel = _sol_attn_int8_kernel if
int8_qk else _sol_attn_kernel`, logged as `int8` or `bf16` (`:222`). So a
bf16-arm against an int8-arm varies the PV dtype, the QK dtype **and** the
implementation at once, and cannot price any one of them. `int8_pv` is passed
only when `int8_qk` is on (`:213`) and **defaults to `True` on the node**
(`:468`) — `SOL_BASELINE_124F` pins it `False`, which is the only reason the
bench's `sage+sol+int8qk` arm is int8-QK with bf16-PV rather than full int8.

### Where the exact work actually goes, by depth

**Measured 2026-08-16** on the 362-frame 1024x768 three-reference capture,
`tau` 1.3, 4 of 56 heads, via `bench/analyze_routing.py`. This is the most
directly actionable table on this page, because unlike the ordering work it
concerns knobs that ship **on**.

| block | diagonal | sink | routed | total exact | `tau` can address |
|---|---|---|---|---|---|
| 0 | 0.2% | 16.6% | 29.8% | **46.6%** | 64% |
| 8 | 0.2% | 16.6% | 12.3% | 29.1% | 42% |
| 24 | 0.2% | 16.6% | 11.3% | 28.1% | 40% |
| 49 | 0.2% | 16.6% | 10.9% | 27.7% | 39% |

Two consequences, both live:

- **The sink is a floor `tau` cannot reach.** 16.6% of all (query block, key
  block) pairs at every depth — 256 sink blocks of 1,539, exactly the
  conditioning share of the sequence. On reference-heavy work only ~39-42% of
  the exact work is `tau`-addressable at depth. Turning `tau` up past the point
  where quality goes has a hard ceiling on what it can buy, and that ceiling is
  set by reference load rather than by the knob.
- **Block 0 is a 1.7x outlier.** It routes 29.8% against ~11% everywhere
  deeper, which is the same shape as `analyze_capture.py`'s finding that early
  attention is near-uniform and leaves a block-sparse router little to exploit.
  `dense_blocks="0"` would cost roughly 1% of total compute **by arithmetic**
  and remove the depth where sparsity is least effective anyway.

**`dense_blocks` and `tau_profile` both ship empty** (`SOL_RECOMMENDED_CUDA`),
so this is unexploited headroom on the shipped path. The `dense_blocks="0"`
figure is derived, not measured end-to-end; it needs a paired render before it
becomes a recommendation.

**The 16.6% is confirmed, by a second path.** It was derived from `video_start`
when first reported; `bench/count_packed_rows.py` now builds the real
`PackedLayout` and reads the segments off it, which reproduces the whole
sequence exactly — 98,524 rows, zero residual — and gives **256 sink blocks of
1,539, 16.6%**. Two independent routes to the same number, one of them the
object the model itself constructs.

The segment breakdown, since nothing else in this repo has ever printed it:

| segment | rows | span |
|---|---|---|
| text | 7,737 | [0..7,737) |
| ref_img | 777 | [7,737..8,514) |
| ref_img | 2,500 | [8,514..11,014) |
| ref_img | 4,128 | [11,014..15,142) |
| audio | 1,206 | [15,142..16,348) |
| video | 82,176 | [16,348..98,524) |

`text` is the one number not counted here — it needs the text encoder, so it is
recovered as the residual against the capture's real sequence length and marked
as such by the tool rather than guessed.

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

**"Pinned exact" is exact for audio and off by one part in a thousand for
visual, and the distinction is worth keeping because this whole argument turns
on those rows being immovable.** The pin is a per-row timestep, not a mask:
`comfy/ldm/minimax/model.py:32-33` sets `VISUAL_COND_TIMESTEP = 0.999` and
`AUDIO_COND_TIMESTEP = 1.0`, applied at `:566-567` as `max(t, aug)` over the
`cond` / `ref_img` and `cond_audio` / `ref_audio` segments. So visual reference
rows carry a deliberate 0.001 of noise augmentation and audio rows are clean.

Verified against a second implementation on 2026-08-16: DiffSynth's H3 pipeline
uses the same formula with the same two constants (`imgvid_cond_noise_aug =
0.999`, `audio_cond_noise_aug = 1.0`, `coderef/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:35-36`). Two
independent sources agreeing makes this one of the better-supported claims on
this page. It changes nothing about Sol — the rows are still pinned and still
unsparsifiable — it just is not "exact".

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

> ### That `exact_kv_and_rows` column is the v1 formula. Recomputed 2026-08-16.
>
> The retraction of this table has existed since 2026-08-16 (`docs/evidence.md`,
> and the "Do not rely on" row above), but **three claims in the prose below it
> were never updated** and went on asserting the withdrawn numbers. Fixed here.
>
> `2*T*S - S²` assumes dense queries over **all** `S` sink rows. v2 of the node
> runs dense queries over the **target audio segment only** — `_sink_blocks`
> returns `sink_q = (audio_start // 64, video_start // 64)`
> (`vendor/sol_attn_minimax.py:487-494`). With `A` = target audio rows, exact
> work is `T*S + A*T - A*S`, which reduces to the old formula exactly when
> `A = S`.
>
> `A` is **measured**, not assumed: `temporal_shape(362)` gives `audio_t` 603,
> and the segment is `audio_t * 2` = **1,206 rows** (`comfy/ldm/minimax/model.py:391`).
>
> | configuration | `exact_kv` | v1 `_and_rows` | **v2 `_and_rows`** |
> |---|---|---|---|
> | t2v, no references | 1.5% | 2.9% | **2.6%** |
> | 3 images at `match` | 4.1% | 8.1% | **5.1%** |
> | 3 images at `max` 1280x720 | 29.6% | 50.5% | **30.2%** |
> | 1 video ref, 124 frames | 17.1% | 31.2% | **17.9%** |
> | 1 video ref, 345 frames | 35.1% | 57.9% | **35.6%** |
>
> **The control that says the method is right:** recomputing the v1 column from
> the doc's own `2p - p²` reproduces all five published figures to within
> rounding. Only then was the v2 formula applied.
>
> **`S` is not reconstructible, and the conclusion does not need it.**
> Attempted 2026-08-16. `A` is measured, the formulae are read from source, and
> `bench/count_packed_rows.py` now makes video and target audio exact — but
> recomputing `S/T` from those plus `docs/h3_references.md`'s measured
> reference costs reproduces the published `exact_kv` column on only **2 of 5**
> rows (1.4% against 1.5%, 29.5% against 29.6%) and misses the video-reference
> rows badly (49.4% against 35.1% at 345 frames). The gap is the text estimate
> at heavy reference load, which needs the encoder. So the `exact_kv` column
> cannot be promoted to measured by arithmetic; that would take a render per
> configuration.
>
> **What makes this survivable is that the finding is insensitive to `p`.** The
> v1 swing is `p - p²` and the v2 swing is `(A/V)(1-p)²` with `A/V` = 0.0112,
> so across the whole plausible range:
>
> | `p` | v1 swing | v2 swing |
> |---|---|---|
> | 17.1% | 14.2 pts | 0.77 pts |
> | 35.1% | 22.8 pts | 0.47 pts |
> | 49.4% | 25.0 pts | 0.29 pts |
>
> **The v2 swing stays under a point wherever `p` actually lands, and shrinks
> as `p` grows** — so if the published column understates `p`, as the
> recomputation suggests, `sink_conditioning` matters *less* than stated here,
> not more. "Not the dominant knob at reference load" holds on any reading.
> Quote the swing; treat the absolute levels as the derived figures they are.

**The mechanism, which is the part worth carrying:** v1 and v2 diverge in
proportion to **how much of the sink is references rather than target audio.**
On t2v the audio segment is most of the sink, so the two agree. At heavy
reference load `A/S` falls to about 2%, the dense-query term nearly vanishes,
and `exact_kv_and_rows` collapses onto `exact_kv`.

**"More context" is not "more sparsity" — but the sink is not what causes it.**
One long video reference forces about **36%** of attention exact, not 58%, and
essentially all of that is the exact-KV side, which `sink_conditioning` cannot
turn off. The reference rows are the cost; running their queries dense was
never the cost.

**`sink_conditioning` is NOT the dominant knob at reference load. Retracted
2026-08-16.** The claim was a 23-point swing at 345 frames. Under v2 the same
swing is **0.5 points** (35.1% to 35.6%), which is smaller than `start_percent`
0.2 → 0.3, smaller than `tau`, and inside any plausible run-to-run spread. The
knob still does what it is for — it keeps the generated audio's queries exact —
but it is a quality knob with a rounding-error price, not a speed lever.

**The v1 path is unreachable on this box. Checked 2026-08-16, having been
guessed at first.** `_sink_blocks` falls back to `sink_q = sink_blocks` when the
layout published no audio span (`vendor/sol_attn_minimax.py:489-491`), and this
page previously said the old column was therefore "stale rather than dead".
That was an inference from reading the fallback, not a check of whether
anything reaches it.

It does not. `PackedLayout` appends the target-audio segment
**unconditionally** — `comfy/ldm/minimax/model.py:390-391`, whose own comment
reads "target audio then target video, always the last two segments" — so
`next(kind == "audio")` always finds one and `audio` is never `None`. Verified
by construction across every shape a shipped graph can take, including
`audio_t=0`, which still yields an `audio` segment of zero rows rather than no
segment:

    t2v 362f            audio_t=603   audio segment present, 1,206 rows
    t2v 124f            audio_t=207   audio segment present,   414 rows
    audio_t forced 0    audio_t=0     audio segment present,     0 rows

So **the v1 column prices nothing that runs here** and the table above is the
whole story for every shipped graph. Two things follow, and the second is why
the fallback should stay:

- The degenerate case is not v1. At `audio_t=0` the audio segment is empty, so
  `sink_q` spans at most one block — `exact_kv_and_rows` collapses onto
  `exact_kv` rather than reverting to the old behaviour. No shipped graph hits
  even this; every one generates audio.
- **The fallback is a guard against an upstream contract change, not dead
  code.** It fires exactly if `PackedLayout` stops appending audio
  unconditionally — which is the class of breakage that hit this repo on
  2026-08-13, when upstream dropped `frame_count` from the same class and every
  graph failed at the Preflight node. Nothing in `bench/` can catch that in
  advance; the fallback is what turns it into degraded pricing instead of a
  crash.

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
video_start // BLOCK]` runs only the audio queries dense.

**This was written as a proposal, and it SHIPPED. Corrected 2026-08-16, having
been stale since 2026-08-14.** The paragraph used to end "the node hardcodes
`sink_q = sink_blocks`. A proposal, not a measurement, and upstream's call."
Upstream took the call: commit `1b675c6` installed v2 (`d856ba83` upstream),
schema-identical to v1, and `_sink_blocks` has returned the narrowed span ever
since (`vendor/sol_attn_minimax.py:487-494`). The cost consequence is in the
recomputed table above, and it is large: this is why `sink_conditioning` stopped
being the dominant knob at reference load.

**The span it narrows to is the right one, verified rather than assumed.**
`_patch_packed_layout` takes `next(kind == "audio")`, and a reference
soundtrack cannot match it: reference audio is labelled `ref_audio`
(`comfy/ldm/minimax/model.py:364`, `:377`) while the target segment is appended
as `audio` immediately before `video` (`:391`, `:398`). So the fallback at
`:489-491` fires only when there is no target audio at all.

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

The `2026-08-15_dense_124f_1344x768` capture this pointed at is **no longer on
disk**. What is there is the 2026-08-17 reference-heavy pair at 362 frames
1024x768, captured Sol-bypassed on Sage fp16 -- real q/k/v, a different geometry,
and carrying references. No kernel has been graded against either. Until that runs, this repo has no accuracy figure to quote
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
- **Morton off**, and the speed argument for that is **RETRACTED 2026-08-16**.
  This bullet used to say the permutation was worth 1.16x alone and cost GPU
  utilisation. That was Triton, at 362 frames, stacked on int8. On the CUDA
  backend the permutation is **free**, measured by a dense control pair that
  varies nothing else: the two pairs disagree in sign (+0.8 s of 861 dense,
  -1.2 s of 454 sparse) and both sit at or under this bench's run-to-run
  spread, so the resolvable cost is zero. Do not quote either delta as a cost;
  that is a mistake `docs/morton.md` made and corrected. The three curves are
  speed-indistinguishable. So there is no longer any speed argument for or
  against Morton, and the case rests entirely on quality, which is unmeasured.
  Upstream defaults it off too, and no H3 profile NVLabs ships reorders at all.
  `Canonical: docs/morton.md`.
- **`sink_conditioning="exact_kv_and_rows"`, on** — but see the reference
  section above, which is where this gets expensive.
- **`dense_blocks`**, which we ship empty and which has two ways to choose a
  list, only one of them currently available. Choosing **our own** needs a
  `SolAttnBlockProbe` run at our own tau, and that instrument went with the
  Triton pack -- see "The Triton node" above; it is not an available action
  today. **Copying a validated list needs no instrument**: every H3 profile
  NVLabs publishes runs `0-1` dense. Costs: seven of fifty blocks was roughly
  54 s on a 362-frame Triton render, and by arithmetic off the 2026-08-16 dense
  pair (860.8 s against 454.0 s over 50 blocks, assuming uniform per-block cost)
  two blocks is about 16 s of 454, or 3.6%. `SOL_RECOMMENDED_CUDA` still ships
  it empty; the decision is in [`docs/roadmap.md`](roadmap.md).

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
| `sink_conditioning` at reference load | **De-prioritised 2026-08-16.** The "23-point swing, biggest lever there is" was v1 arithmetic; recomputed against the v2 node it is ~0.5 points, below the bench's noise floor. Measuring it would now cost a reference-wired bench to resolve a rounding error | still needs reference wiring, but there is no longer a reason to build it for this |
| `start_percent` 0.0–0.4 | zero measurements, ever | none, arms exist |
| `min_tokens` 4096 vs 12288 | our pin is a third of the node's crossover | none |
| re-baseline the frontier above 60k tokens | most numbers here are the wrong regime | GPU hours |
| CUDA e2e vs Triton e2e, ours | we have upstream's 1.4x, not our own | **the Triton pack is deleted**; recover from `kijai/ComfyUI-SolAttn_triton@842c4ea` first |
| **comfy-kitchen's 4090 kernel vs NVLabs' own** | since PR #464 (2026-08-15) there are two independent sm89 implementations; which is faster or more accurate here is unknown, and it is the only external cross-check available on this card | one Python dep (`cutlass.cute`) and a seam -- their API has no `sink_q`, so `exact_kv_and_rows`'s query half needs doing at the integration layer. See [`docs/sol_upstream.md`](sol_upstream.md) |
| **`dense_blocks="0-1"`** | every H3 profile NVLabs ships runs the first two blocks dense and we ship none; it is the one place their tested recipe is strictly more conservative | nothing. It is a config decision, costed in [`docs/roadmap.md`](roadmap.md) |
| quality at tau 1.3, watched to the end | the artifact is temporal and length-dependent | a human watching |
