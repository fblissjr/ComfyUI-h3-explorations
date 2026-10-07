# The frozen-row cache's masked run measured torch's attention kernel, not kitchen's (2026-10-07)

lane: masked
verdict: timed alone on the card, the kitchen int8 kernel runs a rectangle of live queries against every key at the square's rate per pair (11.0 s a step at the masked window's sizes), and torch's own kernel on the same rectangle takes 28.9 s, which is the attention stage the 2026-10-06 run measured (29.2 s); the cached block drops the override that carries the kitchen backend and the checkpoint names none, so that run timed a cache on torch's kernel and its "saves no time" is not a result about the design; nothing was rendered and the retirement stands until the owner reopens it

**What was asked.** The owner, 2026-10-06, of the lane's lead: "make a note
for session tomorrow to verify your conclusions and decide whether there's
anything to salvage from it or if it was close or garbage or whatever". The
note named one unmeasured claim under the result of
[`2026-10-06_frozen_cache_masked_window.md`](2026-10-06_frozen_cache_masked_window.md):
that the attention call for the live queries against every key costs nearly
three times the dense square per query-key pair "on the same kernel". It was
arithmetic from two timed stages, and it named the test: the kernel alone,
on random tensors, at the real sizes.

**How to read it.** One process on an idle card, no model, no server, no
render. Random tensors in the block's layout; one call is one block's
attention and a step is fifty. Each figure is the best of
4 calls after a warm-up; the calls agree to the
third digit and every one is in
[`2026-10-07_frozen_cache_rectangle_kernel.json`](2026-10-07_frozen_cache_rectangle_kernel.json),
which the tables below are printed from.
`bench/bench_rect_attention.py` is the script. 2.14.1+cu132, comfy-kitchen
0.2.37+sol.6272371.up.be003b7, NVIDIA GeForce RTX 4090, bf16. What is **measured** here is the
kernels. What is **read from code** is which one each path reaches. What is
**arithmetic** is marked and is not a result.

## The kernels, alone

The masked window: 113,072 packed rows, of which 28,689 are live on a cached
step (that run's own build line).

| kernel | keys | queries | a call, s | a step of 50 blocks, s | against the square, per pair |
|---|--:|--:|--:|--:|--:|
| kitchen int8, plain | 113,072 | 1,699 | 0.0346 | 1.73 | 2.80 |
| kitchen int8, prequantized | 113,072 | 1,699 | 0.0338 | 1.69 | 2.73 |
| torch sdpa | 113,072 | 1,699 | 0.0410 | 2.05 | not timed on the square |
| kitchen int8, plain | 113,072 | 7,062 | 0.0705 | 3.53 | 1.37 |
| kitchen int8, prequantized | 113,072 | 7,062 | 0.0701 | 3.51 | 1.36 |
| torch sdpa | 113,072 | 7,062 | 0.1461 | 7.30 | not timed on the square |
| kitchen int8, plain | 113,072 | 28,689 | 0.2209 | 11.04 | 1.06 |
| kitchen int8, prequantized | 113,072 | 28,689 | 0.2220 | 11.10 | 1.06 |
| torch sdpa | 113,072 | 28,689 | 0.5775 | 28.87 | not timed on the square |
| kitchen int8, plain | 113,072 | 56,536 | 0.4172 | 20.86 | 1.01 |
| kitchen int8, prequantized | 113,072 | 56,536 | 0.4191 | 20.95 | 1.02 |
| kitchen int8, plain | 113,072 | 113,072 | 0.8235 | 41.17 | 1.00 |
| kitchen int8, prequantized | 113,072 | 113,072 | 0.8247 | 41.24 | 1.00 |

The refine pass's sizes:

| kernel | keys | queries | a step of 50 blocks, s |
|---|--:|--:|--:|
| kitchen int8, plain | 104,515 | 1,699 | 1.56 |
| kitchen int8, prequantized | 104,515 | 1,699 | 1.56 |
| torch sdpa | 104,515 | 1,699 | 1.90 |

- **The kitchen kernel is not slow off the square.** At a quarter of the
  queries it costs the square's rate per pair to within a tenth. A small
  cost per call that does not depend on the queries shows only at the
  smallest rectangles. The two entries cost the same.
- **The square, 41.2 s a step, is what a dense stock
  step's attention was inferred to cost** from the run's stages (the
  session's notes put it near 39.6 s of a 53.4 s step). So the
  stock dense steps are on this kernel.
- **Torch's kernel on the window's rectangle is the stage the run timed**:
  28.9 s here against 29.2 s in
  the timed pass. On the refine pass's sizes it is
  1.90 s against the 2.06 s
  that run's stage split timed, where the kitchen kernel is
  1.56 s.

## Why the cached block reaches torch's kernel (read from code)

- The masked graphs put the kitchen kernel on the model with core's Model
  Attention Backend node, and Sol-Attn chains onto it
  (`workflows/h3_config.py::DENSE_BACKEND_NODE`; node 58 then node 21 in the
  archived probe graph). That node sets
  `transformer_options["optimized_attention_override"]`
  (`comfy/model_patcher.py::set_model_optimized_attention`). The kernel lives
  nowhere else.
- The cached block calls `optimized_attention` with that key removed
  (`_dense_options` in `archive/frozen_cache_masked/frozen_video_cache.py`,
  and unchanged in the live `frozen_video_cache.py`), meaning to step past
  Sol. It steps past the backend too.
- What is left is `preferred_attention=attn.comfy_attention`, whose
  `function` core fills from a `comfy_attention.config` entry in the
  checkpoint (`comfy/ldm/modules/attention.py::ComfyAttention`). The header
  of the checkpoint that graph loads holds no such entry, and neither does
  the refine graphs'. So `function` is `None`.
- The call then runs `optimized_attention` as the module imported it, core's
  default. The server's default mode passes no attention flag
  (`start.sh`, `PERF_ARGS`), so that is `attention_pytorch`.

The evidence that this reading is what happened on the card is the
agreement above: torch's kernel with the two attention stages that were
timed, and the kitchen square with the dense step's attention as it was
inferred. No log line from that run names the kernel.

## What it changes

- **"The cache saves no time" was a result about a cache on torch's
  kernel.** The cost model the run was meant to test assumed the dense
  kernel's time follows the query rows. On the kitchen kernel it does.
- (arithmetic, not measured) Taking the run's plain cached step and
  replacing torch's rectangle with kitchen's gives a cached step near
  18.9 s against the Sol-Attn stock step's 33.6 s, and
  sampling near 267 s against 462.9 s:
  about 42% saved, where the model predicted 40%. It assumes every
  other stage stays as timed, and that the card holds the kitchen kernel's
  working memory beside the model on a cached step, which the dense stock
  steps already do for the larger square.
- **`verify`'s table is confounded.** It set a cached step on exact
  attention against a stock step on Sol's sparse int8 attention. The
  departures it lists hold the cache's stale rows and the difference
  between the two kernels together, and say neither apart.
- **The kept use is on torch's kernel too.** The audio-refine pass's cached
  steps take the same path. There the two kernels are a fraction of a
  second apart per step, the pass is measured to pay as it is, and it is
  left alone: routing it to the kitchen kernel changes its numbers and
  wants its own graded run.
- **The retirement is not reversed here.** The owner retired the masked use
  on yesterday's record. What reopening it would take: the archived module
  in a scratch copy with the cached block's call sent to the dense backend
  the override chain carries; then the same like-for-like pair on the same
  window (`cache_plain` beside `baseline_again`), and `verify` with the
  stock side on dense kitchen for the steps it compares. One window of card
  time. The owner's look at last night's pair is still worth having, with
  the caveat that its cached side is exact attention over stale rows, which
  the corrected cache would not be.

## What this does not show

- No cached step has run on the kitchen kernel. The saving above is
  arithmetic.
- Not established: what the corrected cache's render looks like, its
  departure from stock, or its memory on the card during a cached step.
- Random tensors, not a block's activations. The kitchen kernel's time was
  taken not to depend on the values; that was not tested.
- One card, one build of each library, bf16 only.
- Whether Sol's sparse kernel can take a query subset, which is the larger
  lever yesterday's notes name, is untouched.

## Files

- `2026-10-07_frozen_cache_rectangle_kernel.json`: every call's seconds, the
  environment, the 2026-10-06 run's figures used above, the arithmetic and
  how it was made, and the header counts behind the wiring section.
