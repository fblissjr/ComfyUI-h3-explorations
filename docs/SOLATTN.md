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

**Most of the measurements here were taken in a regime where Sol-Attn cannot
show a gain.** Upstream's own statement is that nothing is visible below
roughly 250–300 frames at 1344x768. The frontier table below is at **length
124 = 37,296 video tokens**, well under that floor. Those are not weak
results; they are measurements of the wrong regime.

What survives: the **362-frame** table (107,856 tokens) and the profiling
table. What does not: the 124-frame frontier numbers, and the quality
judgements taken at that length.

Second standing caveat: **`start_percent` and `end_percent` have never been
measured at all**, at any length, on either backend. They are the paper's
defaults, carried through.

---

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

| canvas | 73 frames | 124 | 250 | 300 | 362 |
|---|---|---|---|---|---|
| 1344x768 (1008/frame) | 22,176 | 37,296 | 72,576 | 87,696 | **107,856** |
| 832x768 (624/frame) | 13,728 | 23,088 | 44,928 | 54,288 | 66,768 |

Upstream's ~100k model ceiling is the 1344x768 / 362-frame / 15s corner. The
floor for seeing anything is ~60k tokens; `bench_e2e_h3.py` warns below it.
**A run under the floor produces a null result that reads as "this knob does
nothing".**

Note this is a *token* floor, not a frame floor — 250 frames is 72,576 tokens
at 1344x768 but only 44,928 at 832x768, on opposite sides of the line.

---

## The CUDA node

`SolAttnMiniMax`, driving `comfy_kitchen.sol_attn`.

### Options

Four have no Triton counterpart; three Triton options are gone, because the
CUDA kernel routes in INT8 unconditionally and there is no quantization choice
left to make.

| option | default | what it does |
|---|---|---|
| `tau` | 1.3 | Routing threshold in sigmas of the proxy row. A key block is exact when its mean score over the query block clears `tau * sqrt(var)`. Higher is sparser. Upstream densities: 1.0 keeps ~16% exact, 1.5 ~7%, 2.0 ~2.7%. |
| `start_percent` | 0.2 | Dense before this point. **Never measured** — see the step table below, it is badly non-linear. |
| `end_percent` | 0.9 | Dense after this point. Also never measured. |
| `min_tokens` | 12288 | Shorter sequences stay dense. `SOL_RECOMMENDED` pins **4096**, which is a third of this and two orders below where gains appear. |
| `sink_conditioning` | `exact_kv_and_rows` | See the reference section — this is the dominant knob at reference load. |
| `morton` | False | Z-order the video tokens so each 64-token block is a compact 3D neighbourhood. Exactly neutral for dense attention. |
| `morton_curve` | `2d_frame` | Z-order within each frame, leaving frame order alone. Correct for H3, whose `FRAME_PER_TOKEN` is `(1,4,4,4,4)`. |
| `centroid_tail` NEW | True | One pooled tail per query block instead of per row, 64x less routing work. Upstream: ~1.4x on the **operation**, **~5–10% end to end**, ~5e-4 cosine. |
| `routed_cap_percent` NEW | 0 | Cap routed blocks as a percent of sequence; 0 is uncapped. Bounds the only workspace term growing with T². Below the actual density it silently degrades routed blocks to their pooled term. |
| `reuse_qkv_memory` NEW | False | Write the output into H3's fused qkv buffer instead of allocating. Upstream: ~1.2 GB at 80k tokens, enough to put attention's peak below the FFN's. Safe for H3, which discards that buffer; leave off for other models. |
| `tau_profile` NEW | unset | Per-block tau, `blocks=tau` separated by `;` or newlines. `force_input`, so it needs a node wired to it. |
| `dense_blocks` | `""` | Blocks kept fully dense, e.g. `0-2,-1`. First and last are the most approximation-sensitive. |
| `verbose` | False | Per-shape dispatch logging, once per distinct shape. |

### What has actually been measured on it here

On a 4090, `B=1 T=512 H=4 D=128` bf16 at tau 1.3, against the eager reference
the algorithm's author wrote:

- **The two backends' arithmetic is equivalent.** CUDA 0.999919 from the
  reference, Triton INT8 0.999885, Triton bf16 0.999995.
- **They run different tail modes.** CUDA defaults `centroid_tail=True`; Triton
  is per-row and has no such parameter. Measured by grading each against both
  reference modes, not read off the source.
- **`reuse_qkv_memory` is numerically identical** to the normal entry — cos
  0.999810 against dense SDPA on a fully-exact control, both to six digits. It
  changes where the output lands, not what it is.

No end-to-end render measurement exists for the CUDA node yet.

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
Load Diffusion Model -> MiniMax H3 SageAttention -> SolAttn* -> BasicGuider
```

Sol must come second: it walks the model's existing object patches and composes
with the attention forwards it finds. Reversed, it overwrites sage's patch and
you silently get sage only.

They **alternate rather than stack.** Inside the sigma window Sol runs sparse
and sage is bypassed; outside it, sage runs dense.

**`ModelAttentionBackend` must not be downstream of either.** ComfyUI's
`set_model_optimized_attention` (`comfy/model_patcher.py:688`) assigns
`optimized_attention_override` unconditionally and its closure discards the
`func` argument — no chaining. Downstream of Sol it deletes Sol silently; the
graph still renders. Sage partly survives because it also object-patches the 50
attention forwards, a path that never reaches `optimized_attention`.

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
