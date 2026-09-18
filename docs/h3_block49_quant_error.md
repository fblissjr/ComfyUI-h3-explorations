# Block 49: why INT8 attention loses accuracy on H3's last blocks, and what has been done about it

Last updated: 2026-09-15, evening (revised end to end after the kernel-side
fix landed in both kernels and three scenes were watched). Written from the sage fork's session at the owner's
request. Model throughout: **MiniMax H3, the pruned int8 convrot fl2va
checkpoint** (`h3_config.MODELS["unet_fl2va"]`), the one the capture set was
rendered with and the one every graph here ships; the weights finding holds
for every H3 DiT checkpoint on disk, see "So what". Capture: the 2026-09-03
base16 t2v set at 1344x768, S=104,361 (395 text, 1,150 audio, 102,816 video
rows), kept to 2026-10-31 (moved from 2026-09-20 on 2026-09-18).

## What "error" means on this page

Not the video. Every render goes through fifty transformer blocks, and
each block runs attention: it takes three tensors called q, k and v (one
row per token, 128 numbers per row per head) and produces one output
tensor of the same shape. That attention step is a fixed piece of
arithmetic with one exact answer for a given q, k, v. A full-precision
kernel (bf16 or fp32 attention) computes that answer to rounding. An
INT8 kernel first rounds q and k to 8-bit integers to go faster, so its
output differs from the exact answer. **The error is the size of that
difference**: take the same q, k, v, run them through the INT8 kernel
and through exact fp32 attention, subtract the two outputs, and divide
the size of the difference by the size of the exact output. That is
"relative L2 error": 0.01 means the INT8 kernel's output is off by one
percent of its own magnitude, on that block, on that step.

**Where the data comes from.** Nothing here is inferred from frames. The
pack's `h3_capture.py` hooks the attention call during a real render and
saves the actual q, k, v the kernel received, for a chosen block and
step (about 4.5 GB per cell at 345 frames; the set on this box covers
blocks 0, 24, 32, 40 and 49 of one text-to-video render). The grade
scripts load a cell, run each kernel on it, run exact attention on it,
and print the numbers in the tables below. Anyone with a card and this
pack can capture their own cells and rerun every script; the results
files under `bench/results/` are those printouts, kept.

**Why it matters that it is measured this way.** The video is the sum of
fifty blocks times sixteen steps of that arithmetic plus a decoder, so a
change you can see in a clip cannot be traced to a block or a kernel by
looking at the clip. The captured-activation error can: it says which
block, which kernel, how much, and whether a change to the kernel moved
it, with the video left out of the loop entirely. The clips come back in
at the end, to ask whether a difference that size is one people notice.

## What is measured here, and why it is not a matter of taste

Every claim on this page except the last row is a number computed from a
file by a script in this repo, on inputs anyone with the file can rerun.
The only opinions are in the last row, and they are labelled as such.

| what | how it is measured | the referent (what "error" is against) | tool | needs |
|---|---|---|---|---|
| Which blocks have loud K channels | energy share of the four largest `k_norm.weight` entries, per block, straight from the checkpoint file | none: a property of the weights | `bench/check_channel_balance.py` (or a five-line one-liner) | the checkpoint, no GPU |
| Whether the activations agree | per-head channel energy of captured q/k/v after norm and RoPE, the channels that dominate, loudest-to-median ratio | none: a property of the tensors | `bench/analyze_head_magnitudes.py` | a capture (`h3_capture.py`, 4.5 GB per cell) |
| How much each INT8 kernel loses | relative L2 of the kernel's output against fp32 attention computed on the same bf16 q/k/v | exact attention on identical inputs, so the only difference is the kernel | `bench/grade_channel_balance.py`, `bench/grade_ck_int8_on_capture.py`, `bench/analyze_sol_error.py` | a capture and the kernels |
| What a fix recovers | the same number, same inputs, same referent, with the fix on | as above; the fix is the only change | same scripts | same |
| That the fix is exact for the math | the fp32 eager reference with and without the rescale: same route, same scores to rounding | the reference against itself | kitchen `tests/test_sol_attn.py -k qk_balance`, sage `tests/test_qk_balance.py` | any CUDA card |
| What it costs | wall time from the server's own history, start to finish, one render at a time | the unmodified render on the same seed | server history | a render |
| Whether people can see it | labelled originals in a fixed order, viewers asked one question, words filed verbatim | none; this row is opinion, three scenes and four viewers of it | `bench/results/2026-09-15_block49_*` | eyes |

The measured rows say the error exists, where, how large, and that the
fix removes part of it at no cost. The last row says whether that shows.
Neither row stands in for the other.

## So what, revised 2026-09-15

**It is not a bug, and it is not ours.** It is a design property of
unrotated INT8 attention meeting this model: SageAttention and
comfy-kitchen's `sol_attn` quantize K with one scale shared across a
row's or a block's 128 channels, which is fine where channels are alike,
and H3's last blocks are not alike -- the released weights put an order of
magnitude of gain on four channels there. Nobody's kernel is wrong; the
model's weights and the quantizer's granularity are a bad match at three
blocks. comfy-kitchen's other INT8 kernel, `int8_attention`, rotates q and
k with a block-Hadamard before quantizing and does not have the problem
(measured, below).

**The blast radius is every H3 user on sparse attention or SageAttention,
and not the rest.** Stock comfy-kitchen's Sol kernel (Comfy-Org's and
kijai's; the per-row K quantizer shares its scale across channels), which
is what ComfyUI core's block-sparse attention node calls; stock
SageAttention upstream and every fork of it, in every mode including the
"accurate" fp16 one (all of them quantize QK to INT8); and by the same
mechanism NVLabs' own INT8 Sol kernels, unmeasured here. NOT
comfy-kitchen's `int8_attention`, the kernel behind ComfyUI core's Model
Attention Backend node on "comfy kitchen attention" and the
`--use-ck-attention` flag: it rotates q/k first, and on the block-49
capture its error is a third of Sol's and of sage's with nothing done
(`bench/results/2026-09-15_ck_int8_attention_block49.json`). So the
common community chain, that backend for the dense steps and the
block-sparse node for the routed ones, carries the problem on its routed
steps only; our own chain, sage on the dense steps, carried it on both,
which is a choice of ours and part of why our failures were as visible
as they were. Every H3 checkpoint variant: the loud channels
are identical across all fourteen full DiT files on this box, and the
turbo/SLA/PDD LoRAs carry no norm weights, so they inherit it (the TaoMate
LoRA too: 208 modules, qkv/out/fc1/fc2 only, read 2026-09-15). Untouched:
anyone on full-precision attention (flash or SDPA in bf16), which has no
scale to share. **That is ComfyUI's default** (read from
`comfy/ldm/modules/attention.py`, 2026-09-15): a stock ComfyUI with the
stock kitchen wheel runs pytorch SDPA unless one of three opt-ins is on,
`--use-sage-attention` (SageAttention's INT8 q/k), the core block-sparse
attention node (kitchen's `sol_attn`), or `--use-ck-attention` (kitchen's
own `int8_attention`). **The third one is immune, measured 2026-09-15**
(`bench/results/2026-09-15_ck_int8_attention_block49.json`,
`bench/grade_ck_int8_on_capture.py`): kitchen's `int8_attention` rotates Q
and K with a block-Hadamard before quantizing, so no channel can dominate
its shared scale; on the block-49 capture its error is a third of Sol's
and a third of sage fp8++'s, and the weights fold moves it by about one
percent. An earlier version of this paragraph listed it as affected by
design; that was inferred from its being a SageAttention port, not read
from its quantizer, and was wrong. The affected population is therefore
everyone on `--use-sage-attention` or the block-sparse node, which is
many of the people rendering long clips and none of the people on
defaults.

**It is a quality effect, not a correctness one, and it is visible.**
Renders complete and are plausible. The error lands on the last block's
sharp read of the text rows (section 5), and that is where it shows: on
three scenes (market, diner at two seeds, restaurant kitchen) every viewer
ranked the unrebalanced default below the rebalanced arms without being
told what to look for, on morphing objects and people, a doubled and
misspelled sign, a chef appearing from nothing (section 6). Not blind,
not a protocol; consistent in direction on every look.

**An outside runtime treats the tail as sensitive too** (read 2026-09-15 by
the TaoMate session, `coderef/TaoMate-H3`): TaoLiveAIGC's TaoMate-H3 runs
H3 with W8A8 on the linears but keeps the first two and last three blocks
in bf16 -- `resolve_h3_cutlass_w8a8_policy` protects blocks 0, 1, 47, 48
and 49, and quantizes only `qkv_proj` and `fc1` of the interior blocks. That
is weight and activation quantization of the linears, with attention in
bf16 FA3, so it is a different surface from the shared-channel-scale
mechanism on this page and corroborates only the sensitivity of the tail,
not the cause. Their protected set and this page's lopsided set (45, 48,
49) overlap on 48 and 49 and disagree on 45 and 47; whether the linear-side
sensitivity and the attention-side one share a root is not established.

**It is fixable where the quantizers are, and this box owns both.** The
identity `q . k == (q * f) . (k / f)` lets K's loud channels be rebalanced
against Q before quantization at no cost to the attention math. Three
forms exist, all off by default as of this writing; "rebalanced" in the
records means all three on:

| lever | where | reaches | block 49 INT8 quantization term (kernel vs fp32 Sol on the same route; Sol's total error moves about a third as much, since the routing term stays) | cost |
|---|---|---|---|---|
| `MiniMaxH3ChannelBalance` (this pack) | per-channel factor from the checkpoint's norm weights, folded into `q_norm`/`k_norm` at load; head-shared, RoPE-pair-equal | any INT8 kernel, ours or not | Sol -13%, sage -7% (8 heads) | none at render time |
| `qk_balance` (sage fork v0.7.19; the Sage node's `fp8++ balanced` mode) | per-head factor from per-call channel norms, inside the per-thread quantizer, gated per head | sage steps | sage -23% (all heads) | +0.7% call, no memory |
| `qk_balance` (kitchen fork `h3-build`, served as `0.2.34+sol.5284cfb`; the Sol node's `qk_balance` widget) | the same per-head factor in Sol's preprocess, inside its pooled, Q and K quantizers, threshold left unbalanced | Sol's routed steps | Sol -27% (8 heads) | one extra read of q per call; none measurable per render |

None reaches the rest: with every lever on, block 49 still sits at
several times block 0, because the block's attention shape amplifies
whatever rounding remains, and a per-channel factor cannot fix a spike
that lives in one token's row. Only finer K scaling inside a kernel
touches that, and on the renders it shows as a small remaining edge for
bf16 attention on the three loud blocks (section 6).

**Where this stands, and what it would take to call it solved:**

1. *Diagnosed.* Closed. Sections 1-5 are the evidence; the checkpoint scan
   and attention-target record is `bench/results/2026-09-14_block49_checkpoint_scan_and_targets.txt`.
2. *Three levers built, measured on captures, served.* Closed for what
   they are. Until the evening of 2026-09-15 all three were off in the
   default graph; that evening the owner moved the default chain to
   kitchen's rotated dense kernel plus Sol with `qk_balance` on, sage out
   (consumer commit 28d8ee5, `docs/wiki/decisions.md`), so one lever is
   now on by default and the other two live in the sage-chain probes.
   A fourth, Sol's own Hadamard rotation (`rotate`), is served since the
   same night and off (`bench/results/2026-09-15_kitchen_0234_rotate_install.json`).
3. *Visible, not yet blind.* Section 6: three scenes, two seeds on one of
   them, four viewers between them, the default ranked last every time on
   prompt-adherence failures (morphs, text). Rebalanced against
   rebalanced-plus-bf16-tail is a close call ("cannot tell them apart",
   "super close, maybe faces", one viewer on hands and coin legibility).
   The blind multi-scene comparison in `docs/SOLATTN.md`'s decision
   standard has not been run; the flip of the default waits on the
   owner's call, with the live h3-repo sessions told first (the freeze session is gone).
4. *Sol's quantizer.* Closed 2026-09-15 evening: the per-head factor is in
   Sol's preprocess on the kitchen fork, graded on captures and installed
   (`bench/results/2026-09-15_kitchen_0234_qk_balance_install.json`).
5. *Open, and not ours alone:* the mechanism, the checkpoint scan and the
   fold numbers are a contribution the kitchen maintainers could act on
   for every user; the sage-side change is one commit anyone forking sage
   could take. Neither has been sent anywhere.
6. *Deeper, not started:* the structural fix inside the kernels. Two
   shapes: finer K scaling (a second scale group for the loud channels),
   or a Hadamard rotation of q and k inside the quantizer, which is what
   kitchen's `int8_attention` already does and which also flattens a
   per-token spike, the case no per-channel rescale can reach. On the
   block-49 capture the rotated kernel beats even the balanced Sol row
   (0.0166 against 0.0193), so rotation is the stronger candidate; Tier 2
   in `docs/h3_quant_policy.md`. Its payoff is the edge bf16 still holds
   over the rebalanced arms, which three scenes put at "small, takes a
   careful look".

## The answer in four sentences

Block 49's `k_norm.weight` puts an order of magnitude more gain on a few
channels than on the rest, so after RMSNorm four K channels carry ~93% of the
block's K energy. Both INT8 attention kernels on these graphs quantize K with one
scale across all 128 channels (sage per 64-token block, Sol per key row), so
those four set the scale and the other 124 keep about three bits. Block 49's attention is also
the peakiest in the model, a handful of effective keys per query on its worst
heads with logits spanning over a hundred, so a small relative error in K
becomes a large logit error and the softmax flips. Loud channels make K's
rounding coarse; peaky attention makes the coarse rounding expensive; the
product is a block whose INT8 error is five times block 0's, almost all of it
on the K side.

Everything below is the evidence for each clause, with where it lives.

## 1. It is the weights, and it is not only block 49

Ranking all fifty blocks of the shipped checkpoint by the energy share of
the four loudest `k_norm.weight` channels (`bench/check_channel_balance.py`
prints it; no card, no capture):

| block | top-4 K-norm energy share | max / median weight |
|---|---|---|
| 49 | 70% | 16x |
| 45 | 31% | 7x |
| 48 | 25% | 6x |
| every other block | 4-6% | 1.0-1.6x |

`q_norm.weight` peaks at the same channels at the same blocks (69% at 49).
The weight peaks at block 49 sit at channels 82 and 19; H3's RoPE is
split-half over channels 0-95 and rotates (i, i+48) together, so each peak
spreads into its mate and the activations show the pairs 34/82 and 19/67.
That is the four-channel set the 2026-08-20 head-magnitude analysis found
(`docs/roadmap.md`, "block 49 attributed, at the input level") and the sage
fork's capture spike reproduced exactly (share 93.2% at block 49, 5-11% at
blocks 0, 32, 40).

Blocks 45 and 48 were never captured. The weights say they carry the same
defect at a third to a half of block 49's strength.

## 2. It is K's rounding, not Q's, not the PV side

A CPU decomposition on the captured block-49 and block-0 cells, fp32 exact
attention on 1,024 sampled query rows (256 text, 256 audio, 512 video) over
all keys and all 56 heads, with each of sage's quantization steps simulated
alone (`tests/spikes/spike_h3_block49_error_anatomy.py` in the sage fork;
K per 64-token block with one scale, Q per token; the fp8 PV side is not
simulated, see the kernel records for that split):

| arm | block 49 | block 0 | 49 / 0 |
|---|---|---|---|
| K int8 alone | 0.0645 | 0.0031 | 21x |
| Q int8 alone | 0.0159 | 0.0052 | 3x |
| Q and K int8 | 0.0666 | 0.0062 | 11x |

K's rounding is essentially the whole QK error at block 49, and it is
twenty times block 0's. Q's is small because its scale is per token and no
single channel dominates a token the way the loud channels dominate a
64-token block of K.

The fp8 PV side, from the kernel records on the same cell
(`spike_h3_real_activations.py`, fp8++ against the fp16 kernel): 0.0472
against 0.0409 at block 49, so the fp8 P and V storage adds about a seventh
on top of the INT8 QK error. Real, and not the mechanism.

## 3. It is amplified by how block 49 attends

Same run, the exact reference's own shape, median over heads and sampled rows:

| | row max p | effective keys | logit range (max minus mean) |
|---|---|---|---|
| block 0, text queries | 0.002 | 25,602 | 7.0 |
| block 0, video queries | 0.003 | 10,653 | 8.9 |
| block 49, text queries | 0.076 | 340 | 15.6 |
| block 49, video queries | 0.095 | 165 | 17.5 |
| block 49, worst-4 heads | 0.453 | 5 | 145.5 |

Block 0 spreads attention over ten thousand keys; block 49 over a few
hundred, and its four worst heads over about five, with logits a hundred
times wider than block 0's. An INT8 error in K is a relative error in the
logit; the same relative error on a logit of 145 moves the softmax by orders
of magnitude more than on a logit of 7. This is why the same quantizer, at
the same granularity, costs five times more at the last block.

The four worst heads by K-rounding error at block 49 are 17, 9, 11 and 22,
each at five to seven times the block's median. Head 11 is the head that
loses under token routing at every step
(`2026-09-04_sol_token_aug_grade.md`), which had no mechanism recorded; a
head that attends to five keys with a hundred-logit range is a candidate.
Hypothesis, not measured.

Text query rows carry the most error at block 49 (0.11 against 0.04 for
video rows). They are sparse queries under the shipped
`sink_conditioning=exact_kv_and_rows`, which runs the audio rows dense and
the text rows not; that is a sparsity-term observation and outside this
page's instrument, noted because the two rankings coincide.

## 4. What can be done, measured

The dot product is invariant under a per-channel rescale of q against k, so
the channels can be rebalanced before quantization at no cost to the math.
Sol's routing threshold is invariant under the same rescale. Measured on the
block-49 cell, sage fp8++ as served, mean rtol against fp32 attention:

| factor form | block 49 | block 0 | where it lives |
|---|---|---|---|
| none | 0.0472 | 0.0085 | |
| per head, from the capture, RoPE-pair-equal (a=0.5) | 0.0381 (-19%) | 0.0088 (+3.5%) | superseded by the in-quantizer form below |
| per channel from the checkpoint's norm weights, pair-equal (a=0.5), CPU simulation of the QK side | 0.0587 vs 0.0666 plain (-12%) | 0.0062 vs 0.0062 (neutral) | `MiniMaxH3ChannelBalance`, folded into the norm weights, free |

**Built 2026-09-15 in the sage fork (v0.7.19, `qk_balance`):** the
per-head factor inside the per-thread INT8 quantizer itself, computed per
call from copy-free channel norms, gated per head on K's loud-channel
share, no RoPE-pair constraint (it acts after RoPE), no calibration, no
extra q/k copy. Kernel-level on the same cells: block 49 0.0472 -> 0.0364
(-22.9%), blocks 40 and 0 unchanged; cost +0.7% on the call at the frame
ceiling and no change in peak memory. Off by default there until a render
check; the record is the fork's CHANGELOG v0.7.19. That is the form to
reach for on the sage steps; this node's weights fold remains the form
that also reaches Sol's kernel, until the same factor is put into Sol's
quantizer.

The alpha sweep put 0.5 at the optimum on both captured block-49 steps;
fully equalizing K (a=1.0) is bad everywhere because Q then carries the whole
scale. The per-channel form gets roughly half the per-head gain because the
loud channels differ in strength across heads and a [128] weight cannot
express that; it is exactly neutral at a flat block, which the per-head form
is not.

**Both kernels share the mechanism, at different strengths.** Read from the
installed kitchen build's source (`comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh`,
`quant_k_rows`, the checkout the wheel was built from): Sol quantizes K per
key row, after subtracting the per-channel key mean, with one INT8 scale
across the row's 128 channels; sage quantizes K per 64-token block with one
scale across the block. Sol's is finer along tokens and has mean-centring
built in, so its loud-channel penalty is smaller than sage's on the same
block, but a row's scale is still set by its loudest channel, so the
rebalancing reaches it. Its routing threshold (eager reference,
`comfy_kitchen/backends/eager/sol_attn.py`: query-block centroid squared
times the per-channel variance of the key-block centroids) is invariant
under the paired rescale in exact arithmetic, so the fold changes which
keys Sol quantizes coarsely, not which blocks it routes.

`MiniMaxH3ChannelBalance` (`channel_balance.py`) is the per-channel form:
two weight patches per block through `ModelPatcher.add_patches`, combo
`balance` off by default, "loud blocks (from weights)" selecting by the
ranking above, "named blocks" taking `dense_blocks` syntax.
`bench/check_channel_balance.py` pins the fold's exactness, its RoPE safety,
the off default and the shipped ranking without a GPU.
`bench/grade_channel_balance.py` is step 3 of the experiment in
`docs/SOLATTN.md`: the Sol kernel's own INT8 term, plain against balanced,
through `analyze_sol_error.py`'s decomposition (kernel called as the node
calls it, since the oracle's own kernel wrapper predates comfy-kitchen#117).
Run 2026-09-14, tau 1.0, first 8 heads, factor from the weights,
`bench/results/2026-09-14_channel_balance_{b49_s15,b0_s15}.json`:

| cell | arm | Sol sparsity_l2 | Sol quant_l2 | Sol total_l2 | sage fp8++ l2 |
|---|---|---|---|---|---|
| block 49, step 15 | plain | 0.0324 | 0.0265 | 0.0415 | 0.0487 |
| block 49, step 15 | balanced | 0.0323 | **0.0231 (-12.9%)** | 0.0393 | 0.0453 |
| block 0, step 15 | plain | 0.1247 | 0.0024 | 0.1244 | 0.0036 |
| block 0, step 15 | balanced | 0.1248 | 0.0024 (+0.9%) | 0.1244 | 0.0036 |

Sol's kernel benefits, by about the same fraction as sage's, and the fold is
neutral at block 0 on both. Routing is unchanged: the eager Sol reference
moves by exactly as much as exact attention does under the bf16 re-rounding
of the balanced inputs (1.69e-2 against 1.67e-2 at block 49, 5.4e-4 against
5.4e-4 at block 0), which is the invariance argument holding in practice.
The sparsity term does not move. So on the shipped stack the fold lowers
the last block's INT8 term for both the dense-window steps and the routed
steps, at no cost, and does nothing anywhere the weights are flat.

## 5. What the peaky heads attend to

Measured on the same two cells, 768 sampled query rows per cell over all
keys, all 56 heads, exact fp32 softmax
(`bench/results/2026-09-14_block49_checkpoint_scan_and_targets.txt`):

| | block 0 | block 49 |
|---|---|---|
| attention mass on the 395 text keys, median head | 0.4% | 12.2% |
| attention mass on the 1,150 audio keys, median head | 17.7% | 16.2% |
| queries whose top-1 key is a text key, median head | 4% | 20% |
| queries sharing one top-1 key (sink signature), median / max head | 2% / 67% | 7% / 34% |
| loudest key row's norm over the median, worst heads | 1.0-1.1x | 1.0-1.4x |

The four worst heads by K-rounding error (17, 9, 11, 22) put 6-38% of
their mass on text keys and their top-1 keys are text tokens (rows 0 and
114) for a tenth to a fifth of queries, with no key-norm outlier. Head 11,
the token-routing loser, is the heaviest text reader of the four at 38%.
So the block-49 error is concentrated on the prompt read, not on a sink
token and not on video-to-video attention.

## 6. Visibility, 2026-09-15: three scenes, and the pair that decides the tail

**On the arm names.** "Shipped" below and in the records means the owner's
default text-to-video graph on this box on this date
(`h3_text_to_video_api.json`: sage `fp8++` plus Sol at the recipe in
`workflows/h3_config.py`), not a released setting and not anyone else's
default; the later records call it `default`.

### 6a. The first look: market scene, one seed, two viewers

The three probe graphs (`h3_text_to_video`, `h3_probe_t2v_balanced`,
`h3_probe_t2v_exact_tail`; same prompt, seed 730451892, 345 frames at
1344x768, 16 steps) rendered on the served build. They were watched as the
original files in that order by an outside viewer and then by the owner,
who has rendered this market scene many times before and knows its habits.
(A first version of this section mapped the outside viewer's notes through
a blind-singles key; the viewer had watched the originals in the listed
order, not the shuffled singles, so two labels were swapped. Corrected the
same day; the blind singles were never scored.)

| arm | what was seen |
|---|---|
| 1, shipped (INT8 attention everywhere) | the porter and the crate morph into something else as he turns. The morph is this scene's known failure: the owner has seen it "often" in prior renders of this prompt |
| 2, balanced (channel-balance node + sage `fp8++ balanced`) | no morph; he goes straight; "way better" than shipped, nothing wrong with it; the one clip where he ends up carrying two crates, arriving with a slight morph-in |
| 3, exact tail (blocks 45, 48, 49 on bf16 attention) | best, "in subtle ways": no morph; he sets the crate down on the table edge and then lifts it, rather than one-handing it, which two viewers had never seen this prompt do; he moves around the shoppers walking toward him instead of through them; the weight shift as he carries is more natural; the audio is louder and crisper; possibly less motion blur |

In all three the woman's identity drifts over the clip; no arm touched
that. The viewers' notes are filed verbatim in
`bench/results/2026-09-15_block49_market_feedback.md`.

The owner's later reading of the morph itself: the worse outputs never
decide whether a crate is one deep box or two shallow trays, and the
morphing is that indecision playing out over time; the ceiling arm commits
to two stacked trays and stays consistent (at the cost of a physically
implausible carry). That is a commitment-to-a-reading effect, which is what
a sharper, less noisy read of the prompt at the last block would produce.

The details that separated the arms are the prompt's own: the weight
shift onto the hip, the coins into the tin, shoppers stepping aside. That
is the prompt-adherence axis section 5 predicts, and the ranking follows
the error sizes: shipped worst, the free levers (which remove a fifth to a
third of the INT8 error on the sage steps and an eighth on Sol's) in the
middle, the ceiling arm (no INT8 error at those blocks on either kernel)
best.

**What this is and is not.** One scene and one seed, so any single
comparison can still be take-to-take luck, and `bench/compare_clip_pixels.py`
confirms the three are three different takes from frame 0. What makes it
more than a coin flip is the prior: the owner's experience of this prompt is
that the porter morphs, and in the two treated arms he does not. The
"never seen before" behaviours in the ceiling arm are the strongest single
claim here and the one most in need of a second seed. The audio difference
is unexplained by the mechanism (the last block's attention reads the
text rows, and audio rows are 1% of the keys); it was measured later the
same day and closed as take-to-take variation (6c).

### 6b. Second scene, two seeds: the diner

The same three arms on `prompt_bank/t2va_diner_breakup.txt` at seeds
730451892 and 20260915 (`bench/results/2026-09-15_block49_diner_batch.md`).
The owner and other viewers, unprompted, on the first seed: the default
last ("a chef morphs out of thin air at the end"; the door sign reads
"TUE SATR DINER" and appears twice), the rebalanced arm second (the sign
legible), the bf16 tail best "in subtle ways". Wall time from the server
history: the rebalanced arm within a second of the default on both seeds,
the bf16 tail about a sixth more. Same ranking as the market scene, so the
direction holds on a second scene and a second seed.

### 6c. The pair that decides the tail: rebalanced vs rebalanced + bf16

With Sol's own factor served (section 8), the open question became whether
the bf16 tail still buys anything once every lever is on. Two graphs
isolate exactly that: `h3_probe_t2v_levers` (balance node + sage balanced
+ Sol balanced, INT8 everywhere) and `h3_probe_t2v_policy` (the same plus
blocks 45/48/49 on bf16). Rendered on the market prompt and on
`prompt_bank/t2va_restaurant_kitchen.txt` at seed 730451892, with the
default alongside (`bench/results/2026-09-15_block49_kitchen_batch.md`);
captioned three-band stacks of every scene exist beside the singles
(`bench/stack_labeled_clips.py`).

Three readings of the market pair, unprompted: the owner could not tell
them apart; one viewer preferred the bf16 tail on "small hands" and
"better coins"; a third called it "super close, maybe faces a tiny bit
less distorted, nothing jumping out". The coin claim was checked on
frames at 0.2 s spacing: both arms render the coin drop; the bf16 arm's
coins are larger, slower and in an open tin, the rebalanced arm's one
small coin is fast and the tin is lidded. A legibility difference, not an
object left undecided (a first reading said "undecided"; corrected in the
record the same evening). Against the unprompted separation of the
default from everything else, this is a much smaller gap.

Measured on the market pair: loudness differs (the bf16 arm about four
and a half LU louder, its peak at clipping), but the six diner clips sit
within two LU of each other with no arm louder, so loudness is the take,
not the tail, and the morning's audio question closes. Motion statistics
are comparable; every frame differs, as any numerics change gives.

**Where that leaves the policy.** Rebalanced is free (wall time within a
second of the default on every scene) and ranked above the default by
every viewer on every scene: the candidate default. The bf16 tail is a
small remaining edge at about a sixth more render time: the opt-in
best-take setting. Closing that edge is kernel granularity work, not more
rescaling (`docs/h3_quant_policy.md`).

## 7. What this does not establish

- A blind verdict. Section 6 is consistent in direction across three
  scenes and four viewers, but every look was on labelled originals in a
  known order; the blind comparison in `docs/SOLATTN.md`'s decision
  standard has not been run.
- Anything about attention paths that are not INT8 with a shared per-row
  scale. fp8 q/k attention gives each element its own exponent, so the
  starving cannot happen the same way, but fp8 still places a per-block
  range with three mantissa bits; expected much smaller, unmeasured. And
  bf16 attention on the loud blocks is the best arm measured, not a
  measured zero: nothing here compares it to fp32.
- Anything outside attention. The loud channels are created by the
  k_norm gain after the qkv projection, so the convrot INT8 linears never
  see them; that is read from the model code, not measured.
- Whether blocks 45 and 48 behave like 49 under balancing. The weights say
  they carry the defect; no capture exists to grade them.
- The remainder. After balancing, block 49 still sits at several times
  block 0, and section 3 says why: the attention shape is the amplifier,
  and no per-channel rescale can fix a spike in one token's row. Finer K
  granularity in the kernel is the lever for that, and it is a kernel
  change on either side, not a weights fold.

## 8. The kernel-side fix, built 2026-09-15: `qk_balance` in the sage fork

### What was done

The weights fold in section 4 is capped by the architecture: the only
per-channel weights after the projection are the q/k RMSNorm gains, which
are shared across heads, so a per-head factor, which the simulation put at
twice the gain, had nowhere to live. The sage fork's per-thread INT8
quantizer is the other place the factor can be applied, and it streams q
and k exactly once, so the multiply is free. Built there as `qk_balance`
(sage fork v0.7.19, its `CHANGELOG.md` and `tests/test_qk_balance.py`):

- Both Triton quant kernels take a factor pointer, `[B, H_kv, C]` fp32,
  and multiply it in right after the load, before the absmax: Q by `f`,
  K by `1/f`. `q . k == (q * f) . (k / f)`, so the attention math is
  unchanged; only the INT8 rounding moves.
- `f = rms_k^0.5 / rms_q^0.5` per (batch, kv head, channel), geometric
  mean one per head, computed per call from `torch.linalg.vector_norm`
  with fp32 accumulation, so no fp32 or bf16 copy of q or k is ever made.
  That is the difference from `smooth_k`, which this stack rejected for
  materializing a K copy at the frame ceiling.
- Gated per head on the energy share of K's four loudest channels; below
  the threshold the head's codes are bit-identical to the plain path.
  Under GQA the factor is per kv head and each query head reads its
  group's.
- No calibration, no capture, no RoPE-pair constraint (it acts after
  RoPE), no per-block list: blocks 45, 48 and 49 open on their own.

### What it measured

Same cells and reference as the rest of this page, sage fp8++ as served:

| cell | plain | `qk_balance` (gate 0.2, the default) | gate 0.5 |
|---|---|---|---|
| block 49, step 15 | 0.0472 | **0.0364 (-22.9%)** | 0.0412 (-12.8%) |
| block 40, step 15 | 0.0426 | 0.0426 (+0.1%) | (no head opens) |
| block 0, step 15 | 0.0085 | 0.0085 (-0.2%) | 0.0085 (+0.0%) |

Per head, in the CPU simulation of the QK side: block 49 loses a third of
its QK error at the default gate and the worst head (17) three-quarters
of its own; block 0 shows a 3% cost in that simulation that the real
kernel's Q rounding and fp8 PV error dilute to nothing, which is why the
default was chosen on the kernel rows. Cost: the two norm passes, +3.3 ms
on the quant step at 104k rows, +0.7% on the whole call, and no change in
peak memory.

For scale against the other numbers on this page: the balanced fp8++ call
at block 49 (0.0364) is below the fp16 kernel's unbalanced error on the
same cell (0.0409), so on this block the fast path with balancing is now
more accurate than the accurate path without it.

### Why it matters

- It removes the identified mechanism at the source, in the code that
  quantizes, rather than working around it in the weights. Every H3
  variant benefits identically because the loud channels are identical
  across them; any other model with late-block outlier channels benefits
  without anyone naming a block.
- It lands the correction on the block that reads the prompt at the
  output head, with nothing after it to absorb error, which is the most
  plausible place for INT8 attention to show up as prompt adherence.
- It costs nothing that this card is short of: no memory, and under a
  percent of the call.
- It is the same idea the LLM quantization world settled on
  (SmoothQuant's migration of difficulty from activations to the other
  operand; KVQuant's per-channel keys), applied to the attention inputs
  of a DiT, in the dynamic per-call form rather than the static
  calibrated one.

### What it covers now, and what is still open

- **Sol's steps: covered since the evening of 2026-09-15.** The same
  factor lives in Sol's preprocess on the kitchen fork (below), so every
  INT8 step on the graph can be balanced: sage's dense window and
  refiner calls, Sol's routed steps.
- **Visible: yes, on three scenes (section 6); blind: not yet.** All three
  levers are still off in the default graph. Turning them on changes
  numerics on every served render, so the flip is the owner's call and
  the live h3-repo sessions are told first (the freeze session is gone).

**Prior art.** The rescale is SmoothQuant's migration pointed at the
attention product instead of a linear layer; what is and is not new in that
is `docs/research/smoothquant_for_attention_qk.md`.

**Sol's quantizer has it too (2026-09-15, evening).** The same per-head
factor, computed in Sol's preprocess from the call's own q/k and applied
inside its pooled, Q and K quantizers with the routing threshold left in
the unbalanced space, is `comfy_kitchen.sol_attn(..., qk_balance=True)` on
the owner's kitchen fork (`h3-build`), exposed by the Sol node's
`qk_balance` widget, off. Graded on the block-49 capture it removes about
twice what the weights fold removes from Sol's INT8 term and is neutral
on blocks 0 and 32 (`bench/results/2026-09-15_channel_balance_kernel_*.json`).
With it, every INT8 step on the graph is balanced. The witness pair
(`h3_probe_t2v_levers` against `h3_probe_t2v_policy`, section 6c) says the
bf16 tail keeps a small edge; the install record is
`bench/results/2026-09-15_kitchen_0234_qk_balance_install.json`, and the
kernel's off path was checked bit-identical to the previous wheel.

## 9. Records

- Sage fork `CHANGELOG.md`: decision log "sm89 q/k quantization" and
  "`smooth_k` on H3: graded across the trajectory"; workload intel "MiniMax
  H3, block 49: per-channel K balancing". Harnesses under `tests/spikes/`
  there: `spike_h3_real_activations.py`, `spike_h3_k_channel_balance.py`,
  `spike_h3_block49_error_anatomy.py`.
- This repo: `docs/roadmap.md` "block 49 attributed, at the input level"
  (2026-08-20) and `bench/results/2026-08-20_head_magnitudes*.json`;
  `docs/SOLATTN.md` "The defaults, re-read against the sage-side error
  records, 2026-09-14"; `bench/results/2026-09-14_channel_balance_*.json`.
- The node on its own, for anyone: `ComfyUI-H3-Quant`
  (https://github.com/fblissjr/ComfyUI-H3-Quant), published 2026-09-15 from
  this repo's `standalone/h3_quant`; this pack defers to it when installed.
- 2026-09-15: `bench/results/2026-09-15_channel_balance_kernel_b{49,0,32}_s15.json`
  (Sol's own factor graded), `2026-09-15_kitchen_0234_qk_balance_install.json`
  (the served wheel), `2026-09-15_block49_market_feedback.md`,
  `2026-09-15_block49_diner_batch.md`, `2026-09-15_block49_kitchen_batch.md`
  (the viewers' words, wall times, loudness, the stacks);
  `docs/h3_quant_policy.md` (the policy and its status log);
  `docs/research/smoothquant_for_attention_qk.md` (prior art).
