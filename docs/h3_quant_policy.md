# H3 quantization policy: which blocks get which precision, and the plan to earn each row

Last updated: 2026-09-15. Owner-approved plan, executing. Model throughout:
MiniMax H3, every DiT checkpoint variant on this box (the finding behind
this page is a base-model property; `docs/h3_block49_quant_error.md`).

## The idea

A quantized H3 is not one setting. It is a per-block policy: for each of the
fifty DiT blocks, what precision the attention kernel quantizes q/k/v to,
and what precision the linears are stored at. Until 2026-09-15 that policy
was "INT8 everywhere" for attention and "int8 convrot everywhere" for the
linears, with the first-two-blocks-dense recipe inherited from upstream and
never measured here. The block-49 work replaced the guess with a mechanism
and a first visible result (`docs/h3_block49_quant_error.md`, sections 1-6):
INT8 attention costs something visible at the last blocks, both free levers
help, and exact attention on the three lopsided blocks helps most.

This page is where the policy lives once each row has evidence. Rows carry a
status; an empty status means nobody has measured it and the row is a
placeholder, not a recommendation.

## The policy table (attention)

| blocks | attention | status | evidence |
|---|---|---|---|
| 0-44, 46, 47 | INT8: kitchen `int8_attention` (q/k rotated before INT8, core's Model Attention Backend) on the dense steps; Sol with `qk_balance` on the routed steps; no sage | shipped 2026-09-15, the default chain (owner) | flat K-norm weights; the kitchen kernel is immune to the loud channels by construction (`bench/results/2026-09-15_ck_int8_attention_block49.json`); Sol's gate shuts per head where nothing is loud (blocks 0 and 32 measured neutral, `bench/results/2026-09-15_channel_balance_kernel_b{0,32}_s15.json`); wall time on one market render per arm (`bench/results/2026-09-15_block49_community_chain.md`) |
| 45, 48, 49 | the same INT8 chain by default; handing them to the kitchen dense kernel (`dense_blocks`, `h3_probe_t2v_ck_dense_tail`) or to bf16 (`MiniMaxH3ExactBlocks`) is the open exception | shipped as INT8; the exception proposed, unscored on this chain | on the sage chain, market and diner 2026-09-15 (`bench/results/2026-09-15_block49_*`): shipped morphs, balanced does not, exact tail best; on the kitchen chain, three arms rendered and unscored (`..._community_chain.md`); Sol's own factor graded on block 49 (`..._kernel_b49_s15.json`) |

## The policy table (linears)

| blocks | linears | status | evidence |
|---|---|---|---|
| all | int8 convrot (`qkv_proj`, `out_proj`, `fc1`, `fc2`) | shipped | inherited; no sensitivity measurement exists here |
| tail | bf16? | unmeasured | TaoMate protects 0, 1, 47, 48, 49 on its W8A8 path; near-miss against our loud set, and a different surface |

## The plan, in tiers

Each tier is usable on its own; each later tier removes a cost of the one
before.

**Tier 0 (config only, today): the policy as a graph.** `h3_probe_t2v_policy`
carries the channel-balance node on the loud blocks, sage in `fp8++
balanced`, and blocks 45/48/49 on exact attention. A few percent of render
time for the full fix at those blocks plus the free levers elsewhere. The
market scene is its regression witness: the porter morphing as he turns is
the visible failure any future change is checked against.

**Tier 1 (days): the free levers reach Sol's steps.** The per-head q/k
balance factor into Sol's own quantizer (`quant_k_rows` / `quant_q_rows`,
computed in the preprocess pass beside the key mean it already takes), so
the routed steps get what the sage steps get. Graded by
`bench/grade_channel_balance.py` on captures; then the witness re-rendered
with balanced everywhere and no bf16 blocks. If that matches exact tail,
the bf16 row disappears and the policy is "levers on."

**Tier 2 (a week): INT8 that survives the loud blocks.** The structural fix
inside both quantizers. First shape considered: mixed-granularity K, the
loud channels permuted into a 16-channel group with its own scale, two
accumulators. Better shape, found 2026-09-15: a Hadamard rotation of q and
k inside the quantizer, as comfy-kitchen's `int8_attention` already does
(randomized block-Hadamard, one scale per token block after rotation);
orthogonal, so exact for the scores, and it flattens per-token spikes too,
which no per-channel factor can. Measured on the block-49 capture that
kernel is immune to the loud channels and beats the balanced Sol row
(`bench/results/2026-09-15_ck_int8_attention_block49.json`). Goal: INT8
at block 49 as good as bf16, so the tail needs no exception and every
user of these kernels gets it.

**Tier 3 (a day, independent): a sensitivity-ranked bake for the linears.**
Capture one step's linear inputs across all fifty blocks, score each
linear's tolerance for 4 bits with the kitchen's own quantizers, bake a
mixed checkpoint with the tail protected on evidence. The VRAM half of a
better quant.

## Today's order (2026-09-15), GPU-sequential, code in parallel

1. Plan page (this), policy graph in the generator. No card.
2. Kitchen branch off `h3-build`: Sol quantizer balance factor. CUDA, then
   build; install only if the grade says yes.
3. When the card frees: the six diner renders (three arms, two seeds),
   scored as they land, originals in listed order.
4. Captures of blocks 45 and 48, so the two blocks the weights flag can be
   graded like 49 was.
5. Grade the rebuilt wheel on captures; re-render the witness with
   balanced-everywhere and no bf16 blocks.
6. Fill this page's rows with what the day earned; flip the defaults it
   justifies, telling the live h3-repo sessions first; tag the served sage build
   if the fork changed.

Not today: Tier 2 and Tier 3.

## Instrumentation this plan needs

- Captures of blocks 45 and 48 (every measurement so far is on 49).
- The audio question: the ceiling arm sounded louder and crisper on one
  seed. Outside the mechanism; if the diner seeds agree it is its own
  thread.
- A second scene at two seeds before any row moves from "proposed" to
  "measured" (the diner batch).

## Status log

- The public pack: https://github.com/fblissjr/ComfyUI-H3-Quant (the
  channel-balance node and the checkpoint scan; the kernel forms go upstream
  through comfy-kitchen).

- 2026-09-15: page created; Tier 0 graph added to the generator; Tier 1
  kernel work started in the kitchen fork.
- 2026-09-15, later: Tier 1 built. Sol's quantizer takes `qk_balance` on the
  kitchen fork branch `h3-qk-balance` (off `h3-build`): the per-head factor
  computed in the preprocess from the call's own q/k, applied inside the
  pooled, Q and K quantizers, threshold and coarse branch left unbalanced.
  Off path bit-identical to the served build on twelve shapes and option
  mixes; the fork's Sol suite gained ten cases (gate per head, shut gate
  reproduces plain bytes, loud channels improve, route invariant, dead rows
  count for nothing). The Sol node carries the switch as `qk_balance`, off;
  the policy graph and a new all-levers witness (`h3_probe_t2v_levers`)
  turn it on. Not yet graded on captures (the card was rendering the diner
  batch); not installed; not merged to `h3-build`.
- 2026-09-15, evening: Tier 1 graded and installed. On the block-49 capture
  Sol's own factor removes about twice what the weights fold removes from
  its INT8 term, and it is neutral on blocks 0 and 32 where the gate is
  shut (`bench/results/2026-09-15_channel_balance_kernel_b{49,0,32}_s15.json`;
  the fork's Sol suite on the wheel: only the pre-existing top-k ties case
  fails). Diner batch at two seeds replicated the market ranking
  (`bench/results/2026-09-15_block49_diner_batch.md`), and the two free
  levers cost no measurable wall time. `h3-build` fast-forwarded to the
  branch, wheel installed through `vendor/rebuild_kernel.sh`, server
  restarted, graphs regenerated. Next: the all-levers witness against
  exact tail; if it matches, the bf16 row goes.
- 2026-09-15, later still: the witnesses rendered on the served
  `0.2.34+sol.5284cfb`, market prompt, seed 730451892, the same seed as the
  morning's three arms: `Video/h3_probe_t2v_levers_00001-audio.mp4` (every
  free lever, no bf16 blocks; wall time within a second of the shipped
  render, server history) and `Video/h3_probe_t2v_policy_00001-audio.mp4`
  (the same plus blocks 45/48/49 exact; wall time within a second of the
  exact-tail render). Unscored at the time of writing; the question is
  whether `levers` matches `exact_tail` on the porter and the crate. If it
  does, the bf16 row above goes and the policy is "levers on".
- 2026-09-15, evening: the owner is showing the Tier 1 pair to viewers as
  clip 1 = `h3_probe_t2v_levers_00001-audio.mp4` (every free lever, INT8
  everywhere) and clip 2 = `h3_probe_t2v_policy_00001-audio.mp4` (the same
  plus blocks 45/48/49 on bf16), originals in that order, one question:
  which is better, or are they the same. The pair isolates exactly one
  change, the bf16 tail on top of the levers. "Same" or a preference for
  clip 1 removes the bf16 row; a preference for clip 2 keeps it at about a
  sixth more wall time.
- 2026-09-15, evening: the market pair read three ways (cannot tell apart;
  clip 2 on hands and coins, the coins verified as a legibility difference;
  "super close, maybe faces"), a much smaller gap than the morning's
  shipped-vs-rest look (`bench/results/2026-09-15_block49_market_feedback.md`).
  Second scene rendered for the same pair, restaurant-kitchen prompt, seed
  730451892: `Video/block49_kitchen/levers_s730451892_00001-audio.mp4` and
  `policy_s730451892_00001-audio.mp4` (wall times in the server history:
  levers at the shipped cost, policy about a sixth more, as before).
  Unscored. Direction the evidence points: levers on by default (free,
  always ranked above shipped), the bf16 tail as the opt-in best-take
  setting, Tier 2 the only way to retire it. No default flipped yet; the
  live h3-repo sessions are told before that lands (freeze is gone).
- 2026-09-15, late: kitchen's own `int8_attention` (ComfyUI's
  `--use-ck-attention`) graded on the block-49 and block-0 captures: immune
  to the loud channels by construction (Hadamard rotation of q/k before
  INT8), a third of Sol's and sage's error on block 49, the fold moves it
  one percent (`bench/results/2026-09-15_ck_int8_attention_block49.json`).
  Two consequences: the blast radius is sage and Sol, not every INT8
  attention; and Tier 2's shape is rotation, not group scales. Not timed;
  whether that kernel is a viable default at H3 length is a speed question
  nobody here has asked yet.
- 2026-09-15, late: the community chain rendered on the market prompt
  (`bench/results/2026-09-15_block49_community_chain.md`): kitchen's rotated
  INT8 kernel on the dense steps plus Sol, three arms (as run, with Sol's
  balance, with 45/48/49 on the dense kernel). Same wall time as our
  sage-dense chain; the dense tail costs 16 s here, not 60, because the
  fallback is INT8 rotated rather than bf16. Unscored. Two things this
  changes regardless of the scoring: our Sage node is a candidate for
  removal from the default graph (same speed, a third of the error on
  block 49, measured), and Tier 2 is rotation.
- 2026-09-15, night: the default flipped (owner). Every video graph's
  dense kernel under Sol is core's Model Attention Backend on kitchen int8,
  sage off, and Sol's `qk_balance` is on (`h3_config.DENSE_BACKEND_NODE`,
  `h3_config.SOL_RECOMMENDED_CUDA`; `docs/wiki/decisions.md`). The sage-chain
  arms `h3_probe_t2v_levers`, `h3_probe_t2v_policy` and
  `h3_probe_t2v_exact_tail` stay on sage so their pair and ceiling still mean
  what they did; `h3_probe_t2v_balanced` and `h3_probe_t2v_ck_balanced`
  retired (the second is the default now). `h3_probe_t2v_ck` is the
  community chain as most people run it, Sol's balance off. The
  community-chain arms are still unscored, so the tail row stays open on
  this chain.

## Tier 2 design note (2026-09-15, evening): rotation inside Sol's quantizer

What comfy-kitchen's `int8_attention` does, done in `sol_attn`'s
preprocess, so the routed steps stop needing a rebalance or a dense tail:

- Rotate every q row and every k row by the same randomized H128 before
  quantizing: kitchen's own device helpers (`apply_convrot_sign128`,
  `convrot128` in `quant_qk_int8.cu`: fixed sign diagonal, then a
  normalized Walsh-Hadamard over the 128 channels via warp shuffles) are
  the implementation to lift. Orthogonal, same signs on both sides, so
  every exact-branch score is unchanged; the INT8 tiles read rotated bytes
  and need no change.
- Apply it in `quant_q_rows`, `quant_k_rows` (after the key-mean
  centring), `centroid_quant` and `prep_pooled_quant`, i.e. everywhere a
  row becomes int8. The routing threshold, kmean, kcvar and the coarse
  branch stay unrotated, exactly as `qk_balance` left them: they are read
  against each other, never against an int8 carrier.
- Layout: the per-row work in `quant_k_rows` is one thread per row over
  128 channels; the Hadamard wants one warp per row (four channels per
  lane). That is the one real change of shape, and the reason this is a
  day and not an hour.
- Expected result on the block-49 capture: at or below the rotated
  kitchen kernel's row (0.0166), against balanced Sol's 0.0193 and plain
  Sol's 0.0265, with per-token spikes handled too. `qk_balance` becomes
  redundant with rotation on and can stay as it is.
- Off by default behind a `rotate` flag until graded; the off path stays
  bit-identical. Same grading path as Tier 1
  (`bench/grade_channel_balance.py` gains a `rotated` row).
- 2026-09-15, night: Tier 2 built and graded, not installed. Kitchen fork
  branch `h3-sol-rotate` (off `h3-build`): `sol_attn(..., rotate=True)`, a
  fixed sign-diagonal-plus-Hadamard rotation of every q/k row before Sol's
  INT8 quantizers, threshold unrotated, off by default and byte-identical
  off. On the block-49 capture, Sol's INT8 term: plain 0.0265, balanced
  0.0193, rotated 0.0140, both 0.0126; block 0 improves slightly too
  (`bench/results/2026-09-15_sol_rotate_b{49,0}_s15.json`). That is below
  kitchen's rotated dense kernel's 0.0166 on the same cell. The fork's Sol
  suite from the wheel: 128 passed, the pre-existing top-k ties case only.
  Install, a Sol node `rotate` widget and a witness render come after the
  default flip devguy is making lands, so the two changes do not cross in
  the generator.
- 2026-09-15, night: the Tier 2 witness rendered on the served rotate wheel:
  `Video/h3_probe_t2v_rotate_00001-audio.mp4` (the default chain with Sol's
  rotation on, market prompt, seed 730451892; wall time from the server
  history 534 s against 500 s for the same chain without it, the first
  render after a restart). Its pair is `h3_probe_t2v_ck_balanced_00001`,
  which is the new default chain exactly (kitchen dense + Sol qk_balance),
  rendered earlier today on the same seed. Unscored.
- 2026-09-15, late night: the route and the cost measured
  (`bench/results/2026-09-15_sol_route_b{49,0}_s15.json`,
  `2026-09-15_sol_timing_b49_s15.json`; scripts
  `bench/grade_sol_route_on_capture.py`, `bench/time_sol_options_on_capture.py`).
  Route: against the fp32 route recomputed with the eager rule, the INT8
  kernel's routed-block count per query block is off by about two blocks on
  block 49 and about one on block 0 (counts only; a swap at equal count
  reads as agreement); the balance takes a fifth off that at block 49,
  rotation a quarter, both together two fifths, and at block 0 rotation
  alone takes a third off. Density is unchanged in every arm, so the error
  is scatter, not bias. That is the first number for "make Sol's routing
  better", and rotation moves it on every block, not only the loud ones.
  Cost: qk_balance one to two percent of a Sol call, rotate about fifteen
  (serial per-row Hadamard; a warp-per-row form is the optimization if it
  earns a default). Two control renders queued for the coins question
  (`h3_probe_t2v_dense`, `h3_probe_t2v_sol_nosage`, market seed).
- 2026-09-15, evening, the coins: the two controls say the flaw all three
  community-chain arms share (coins with no hand) belongs to the kitchen
  rotated INT8 dense kernel on the early steps, not to Sol and not to the
  take: bf16 dense with Sol and bf16 dense without Sol both keep the hand,
  as do sage's dense steps (`bench/results/2026-09-15_block49_community_chain.md`,
  the controls section). One seed. Consequence for the policy: the dense
  kernel is not settled by the block-49 last-step grade alone; the
  early-step and mid-block grades of the three dense kernels come next,
  and the sage chain with every lever plus Sol rotate is rendering as the
  comparison arm.
- 2026-09-15, evening: `h3_probe_t2v_sage_rotate` rendered (the sage chain
  with every lever plus Sol's rotation; 508 s, the same cost as the plain
  sage chain): the hand does the coin drop. Four dense-step configurations
  keep the hand (sage, sage with every lever and rotation, bf16 with Sol,
  bf16 alone) and the three that lose it all run kitchen's rotated INT8
  dense kernel. One seed, but the split is clean along one variable.
  Files for the owner's look: `Video/h3_probe_t2v_sage_rotate_00001-audio.mp4`
  against `Video/h3_probe_t2v_ck_balanced_00001-audio.mp4` (the current
  default) and `Video/h3_probe_t2v_sol_nosage_00001-audio.mp4` (bf16
  dense + Sol).
- 2026-09-15, night: the early-step grade of the three dense kernels
  (`bench/results/2026-09-15_dense_kernels_by_step.json`) puts kitchen's
  rotated INT8 kernel below sage at every cell, step 4 included; the
  coins flaw is not a per-call-error effect this instrument can see. One
  seed of clips against a lower number everywhere: the default dense
  kernel stays an open question, to be settled by the same pair on a
  second seed and a second scene. Token routing (`bench/results/2026-09-15_sol_token_aug_x_options_b49_s15.json`):
  on the fixed producer branch, balance + rotation + token routing is the
  best Sol number on block 49; token routing without the balance still
  hurts it.
- 2026-09-15, end of session: eleven reproduction renders queued under
  `Video/block49_repro/`, default chain (kitchen dense + Sol balanced)
  against the sage levers chain, same seed per pair: diner, the kitchen
  scene, market at seed 20260915, hardware aisle, post office, noodle bar.
  Filenames `<scene>_default_s<seed>_00001-audio.mp4` and
  `<scene>_sagelevers_s<seed>_00001-audio.mp4`. Every capture instrument
  says kitchen's kernel is the better one (`bench/results/2026-09-15_dense_kernels_*.json`);
  these clips decide whether the market seed was luck. Postmortem in the
  sage fork's internal postmortems, 2026-09-15.
- 2026-09-15, later: all eleven reproduction renders succeeded; wall
  times and the scoring table in
  `bench/results/2026-09-15_block49_repro_batch.md`; captioned
  default-over-sagelevers stacks per scene under
  `Video/block49_repro/stacks/`. Scoring is the owner's, pending.
